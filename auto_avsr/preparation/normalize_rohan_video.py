#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-normalise the preprocessed ROHAN videos to match the visual front-end.

The first ROHAN preprocessing copied the raw 300x300 LFROI videos through
untouched at their native ~30 fps. That is a double mismatch against the
English visual front-end (and the Conformer back-end), which were trained on
LRS3 mouth ROIs at **25 fps** and a mouth-centred scale (a ~96 px ROI that the
``CenterCrop(88)`` transform trims to 88 px). Feeding a 300x300 ROI to
``CenterCrop(88)`` yields a pathologically over-zoomed lip patch (only the lips,
philtrum/chin cut off), and 30 fps skews the temporal dynamics by ~20 %.

This pass reads the EXISTING preprocessed dataset (whose audio + Katakana
transcripts are already correct) and rewrites only the video stream:

  * temporal:  resample frames from the source fps to ``--target-fps`` (25);
  * spatial:   centre-crop each frame to ``--central`` px, then resize to
               ``--mouth-size`` px (default 200 -> 96), so the downstream
               ``CenterCrop(88)`` sees a properly-scaled, mouth-centred ROI.

Audio (.wav, already 16 kHz) is copied alongside unchanged, and the label CSVs
are rewritten with the new (25 fps) frame counts. The train/val/test split is
inherited verbatim from the source labels, so it is unchanged.

At 25 fps the dataloader's ``rate_ratio`` of 640 (= 16000 / 25) is now also
correct for AVSR audio/video sync (it was wrong at 30 fps).

Example::

    python preparation/normalize_rohan_video.py \
        --src ../datasets/rohan_preprocessed \
        --dst ../datasets/rohan_preprocessed_fixed \
        --target-fps 25 --central 200 --mouth-size 96
"""

import argparse
import glob
import os
import shutil

import av
import numpy as np
import torch
import torchvision


def load_video_thwc(path):
    """Decode a video to (uint8 T x H x W x C RGB, source_fps)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        src_fps = float(stream.average_rate) if stream.average_rate else 25.0
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not frames:
        return torch.empty((0, 0, 0, 3), dtype=torch.uint8), src_fps
    return torch.from_numpy(np.stack(frames)), src_fps


def resample_indices(n_src, src_fps, target_fps):
    """Nearest-frame indices mapping src frames onto a target-fps timeline."""
    if n_src == 0:
        return []
    n_dst = max(1, int(round(n_src * target_fps / src_fps)))
    return [min(n_src - 1, int(round(k * src_fps / target_fps))) for k in range(n_dst)]


def normalize_frames(vid_thwc, central, mouth_size):
    """Centre-crop each frame to ``central`` px then resize to ``mouth_size`` px.

    Input/-output are uint8 RGB. rtype: uint8 T x mouth_size x mouth_size x C.
    """
    x = vid_thwc.permute(0, 3, 1, 2)  # T,C,H,W uint8
    x = torchvision.transforms.CenterCrop(central)(x)
    x = torchvision.transforms.Resize(
        mouth_size, antialias=True,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )(x)
    return x.permute(0, 2, 3, 1).contiguous()  # T,H,W,C


def encode_video(path, frames_thwc, fps):
    """Write a uint8 T x H x W x C RGB tensor to an H.264 mp4 via PyAV."""
    frames = frames_thwc.numpy()
    height, width = int(frames.shape[1]), int(frames.shape[2])
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():  # flush
            container.mux(packet)


def process_split(csv_path, src_root, dst_root, dst_seg, args):
    rows_out = []
    rows = [r for r in open(csv_path, encoding="utf-8").read().splitlines() if r.strip()]
    for i, row in enumerate(rows):
        dataset_name, rel_path, _old_len, token_ids = row.split(",", 3)
        src_mp4 = os.path.join(src_root, dataset_name, rel_path)
        vid, src_fps = load_video_thwc(src_mp4)
        idx = resample_indices(len(vid), src_fps, args.target_fps)
        vid = vid[idx]                                   # temporal resample
        vid = normalize_frames(vid, args.central, args.mouth_size)  # spatial

        dst_mp4 = os.path.join(dst_seg, os.path.basename(rel_path))
        encode_video(dst_mp4, vid, args.target_fps)

        # carry the (already-correct) audio across unchanged
        src_wav = src_mp4[:-4] + ".wav"
        if os.path.isfile(src_wav):
            shutil.copyfile(src_wav, dst_mp4[:-4] + ".wav")

        rows_out.append(f"{dataset_name},{rel_path},{len(vid)},{token_ids}")
        if (i + 1) % 100 == 0:
            print(f"  [{os.path.basename(csv_path)}] {i + 1}/{len(rows)}", flush=True)
    return rows_out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="existing rohan_preprocessed root_dir")
    ap.add_argument("--dst", required=True, help="output root_dir for the fixed dataset")
    ap.add_argument("--target-fps", type=float, default=25.0)
    ap.add_argument("--central", type=int, default=200,
                    help="centre-crop size (px) applied to the source ROI before resize")
    ap.add_argument("--mouth-size", type=int, default=96,
                    help="output frame size (px); CenterCrop(88) trims it downstream")
    ap.add_argument("--limit-csv", default=None,
                    help="process only this single CSV filename (for a quick smoke test)")
    args = ap.parse_args()

    src_labels = os.path.join(args.src, "labels")
    dst_labels = os.path.join(args.dst, "labels")
    # video segments live under <root>/rohan/rohan_video_seg per the source layout.
    rel_seg = os.path.join("rohan", "rohan_video_seg")
    dst_seg = os.path.join(args.dst, rel_seg)
    os.makedirs(dst_labels, exist_ok=True)
    os.makedirs(dst_seg, exist_ok=True)

    csvs = sorted(glob.glob(os.path.join(src_labels, "*_transcript_lengths.csv")))
    if args.limit_csv:
        csvs = [c for c in csvs if os.path.basename(c) == args.limit_csv]
    print(f"[info] normalising {len(csvs)} split(s): "
          f"fps->{args.target_fps}, central {args.central}px -> {args.mouth_size}px", flush=True)
    for csv_path in csvs:
        rows = process_split(csv_path, args.src, args.dst, dst_seg, args)
        out_csv = os.path.join(dst_labels, os.path.basename(csv_path))
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"[done] {os.path.basename(csv_path)}: {len(rows)} rows -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
