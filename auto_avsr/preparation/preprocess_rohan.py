#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocess the Japanese ROHAN4600 multimodal corpus into the Auto-AVSR format.

The corpus (``datasets/ROHAN4600``) is laid out as three sibling folders::

    ROHAN4600/
    ├── ROHAN4600_zumndamon_normal_label/
    │       ROHAN4600_0001.lab ...                      # HTK phoneme alignment
    ├── ROHAN4600_zumndamon_normal_picture/
    │       ROHAN4600_0001-0400_LFROI/LFROI_ROHAN4600_0001.mp4 ...  # lip ROI video
    │       ROHAN4600_0001-0400_csv/ROHAN4600_0001_points.csv ...   # landmarks (unused)
    └── ROHAN4600_zumndamon_normal_synchronized_wav/
            ROHAN4600_0001.wav ...                      # 96 kHz mono audio

Unlike LRS2/LRS3, the videos are **already mouth/lip ROIs** (the ``LFROI``
crops), so we do NOT run face detection / mouth cropping -- we copy the ROI
video straight through. The audio is resampled to the model's rate (16 kHz by
default) and the phoneme labels are converted to Katakana
(``rohan_phonemes.lab_to_katakana``).

Output layout (consumed by ``datamodule/av_dataset.py``)::

    out_dir/
    ├── rohan/
    │   ├── rohan_video_seg/<id>.mp4   (+ <id>.wav alongside)
    │   └── rohan_text_seg/<id>.txt
    └── labels/
        ├── rohan_train_transcript_lengths.csv
        ├── rohan_val_transcript_lengths.csv
        └── rohan_test_transcript_lengths.csv

Each CSV row: ``rohan,<rel_path>.mp4,<num_video_frames>,<space-separated token ids>``.

Example::

    python preprocess_rohan.py \
        --rohan-root ../../datasets/ROHAN4600 \
        --out-dir    ../../datasets/rohan_preprocessed \
        --audio-sample-rate 16000
"""

import argparse
import glob
import os
import random
import re
import shutil
import sys

import av
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchvision

# preparation/ lives under auto_avsr/, so make the repo root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datamodule.transforms import JapaneseTextTransform  # noqa: E402
from preparation.rohan_phonemes import lab_to_katakana  # noqa: E402
from preparation.utils import save2aud  # noqa: E402

DATASET_NAME = "rohan"

# Default sub-folder names inside the raw ROHAN4600 corpus.
LABEL_SUBDIR = "ROHAN4600_zumndamon_normal_label"
PICTURE_SUBDIR = "ROHAN4600_zumndamon_normal_picture"
WAV_SUBDIR = "ROHAN4600_zumndamon_normal_synchronized_wav"

# "<corpus>_<start>-<end>_LFROI"
_LFROI_DIR_RE = re.compile(r"_(\d+)-(\d+)_LFROI$")


def find_utterance_ids(label_dir):
    """Return sorted utterance ids (e.g. ``ROHAN4600_0001``) from ``*.lab``."""
    ids = []
    for lab in glob.glob(os.path.join(label_dir, "*.lab")):
        ids.append(os.path.splitext(os.path.basename(lab))[0])
    return sorted(ids)


def build_video_index(picture_dir):
    """Map utterance id -> LFROI mp4 path by scanning the batched ROI folders."""
    index = {}
    for entry in sorted(os.listdir(picture_dir)):
        full = os.path.join(picture_dir, entry)
        if not os.path.isdir(full) or not _LFROI_DIR_RE.search(entry):
            continue
        for mp4 in glob.glob(os.path.join(full, "LFROI_*.mp4")):
            # LFROI_ROHAN4600_0001.mp4 -> ROHAN4600_0001
            utt_id = os.path.splitext(os.path.basename(mp4))[0][len("LFROI_"):]
            index[utt_id] = mp4
    return index


def split_ids(ids, val_size, test_size, seed, shuffle):
    """Partition ids into (train, val, test) deterministically."""
    ids = list(ids)
    if shuffle:
        random.Random(seed).shuffle(ids)
    n = len(ids)
    if val_size + test_size >= n:
        raise ValueError(
            f"val ({val_size}) + test ({test_size}) >= total utterances ({n})."
        )
    test = ids[: test_size]
    val = ids[test_size : test_size + val_size]
    train = ids[test_size + val_size :]
    # Sort within each split for stable, readable CSVs.
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def decode_video_thwc(path):
    """Decode every frame of a video as a uint8 ``T x H x W x C`` (RGB) tensor.

    torchvision >= 0.20 dropped ``torchvision.io.read_video``; decode with PyAV.
    """
    with av.open(str(path)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if not frames:
        return torch.empty((0, 0, 0, 3), dtype=torch.uint8)
    return torch.from_numpy(np.stack(frames))


def encode_video(path, frames_thwc, fps):
    """Write a uint8 ``T x H x W x C`` (RGB) tensor to an H.264 mp4 via PyAV."""
    frames = frames_thwc.numpy() if hasattr(frames_thwc, "numpy") else frames_thwc
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


def video_frame_count(path):
    """Number of frames in a video (drives length-based batching)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:  # container metadata, when available
            return stream.frames
        return sum(1 for _ in container.decode(video=0))


def load_resampled_audio(wav_path, target_sample_rate):
    """Load a wav, downmix to mono and resample to ``target_sample_rate``."""
    # torchaudio >= 2.9 routes load() through torchcodec; read via soundfile.
    data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))  # channels x frames
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, target_sample_rate
        )
    return waveform


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_root = os.path.normpath(os.path.join(repo_root, "..", "datasets", "ROHAN4600"))
    default_out = os.path.normpath(os.path.join(repo_root, "..", "datasets", "rohan_preprocessed"))

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rohan-root", default=default_root,
                        help=f"Raw ROHAN4600 corpus root (default: {default_root}).")
    parser.add_argument("--out-dir", default=default_out,
                        help=f"Output preprocessed root_dir (default: {default_out}).")
    parser.add_argument("--audio-sample-rate", type=int, default=16000,
                        help="16000 for ASR/VSR; 19200 for the paper's AVSR setting.")
    parser.add_argument("--video-fps", type=int, default=25,
                        help="Recorded for metadata; the ROI videos are copied as-is.")
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Split by sorted id order instead of a seeded shuffle.")
    parser.add_argument("--reencode", action="store_true",
                        help="Re-encode ROI videos via torchvision instead of copying.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N utterances (0 = all; for quick tests).")
    args = parser.parse_args()

    label_dir = os.path.join(args.rohan_root, LABEL_SUBDIR)
    picture_dir = os.path.join(args.rohan_root, PICTURE_SUBDIR)
    wav_dir = os.path.join(args.rohan_root, WAV_SUBDIR)
    for d in (label_dir, picture_dir, wav_dir):
        if not os.path.isdir(d):
            parser.error(f"expected corpus sub-folder not found: {d}")

    text_transform = JapaneseTextTransform()
    video_index = build_video_index(picture_dir)

    # Keep only utterances that have label + video + audio.
    all_ids = find_utterance_ids(label_dir)
    usable, missing = [], []
    for utt_id in all_ids:
        wav_path = os.path.join(wav_dir, f"{utt_id}.wav")
        if utt_id in video_index and os.path.isfile(wav_path):
            usable.append(utt_id)
        else:
            missing.append(utt_id)
    if missing:
        print(f"[warn] skipping {len(missing)} utterances missing video/audio "
              f"(e.g. {missing[:3]})")
    if args.limit:
        usable = usable[: args.limit]
    print(f"[info] {len(usable)} usable utterances out of {len(all_ids)} labels")

    splits = split_ids(usable, args.val_size, args.test_size, args.seed,
                        shuffle=not args.no_shuffle)

    labels_dir = os.path.join(args.out_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    rel_dir = f"{DATASET_NAME}_video_seg"
    seg_dir = os.path.join(args.out_dir, DATASET_NAME, rel_dir)
    text_seg_dir = os.path.join(args.out_dir, DATASET_NAME, f"{DATASET_NAME}_text_seg")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(text_seg_dir, exist_ok=True)

    unknown_tokens = []
    for subset, ids in splits.items():
        csv_path = os.path.join(labels_dir, f"{DATASET_NAME}_{subset}_transcript_lengths.csv")
        rows = []
        for i, utt_id in enumerate(ids):
            lab_path = os.path.join(label_dir, f"{utt_id}.lab")
            src_video = video_index[utt_id]
            src_wav = os.path.join(wav_dir, f"{utt_id}.wav")

            transcript = lab_to_katakana(lab_path, unknown=unknown_tokens)
            if not transcript:
                print(f"[warn] empty transcript for {utt_id}, skipping")
                continue

            out_video = os.path.join(seg_dir, f"{utt_id}.mp4")
            out_audio = os.path.join(seg_dir, f"{utt_id}.wav")
            out_text = os.path.join(text_seg_dir, f"{utt_id}.txt")

            # -- video: copy the lip ROI as-is (or re-encode on request).
            if args.reencode:
                vid = decode_video_thwc(src_video)
                encode_video(out_video, vid, args.video_fps)
                num_frames = vid.shape[0]
            else:
                shutil.copyfile(src_video, out_video)
                num_frames = video_frame_count(out_video)

            # -- audio: resample 96 kHz mono -> target rate, save alongside.
            waveform = load_resampled_audio(src_wav, args.audio_sample_rate)
            save2aud(out_audio, waveform, args.audio_sample_rate)

            # -- text + tokens.
            with open(out_text, "w", encoding="utf-8") as f:
                f.write(transcript)
            token_id = text_transform.tokenize(transcript)
            token_id_str = " ".join(map(str, token_id.tolist()))

            rel_path = os.path.join(rel_dir, f"{utt_id}.mp4").replace(os.sep, "/")
            rows.append(f"{DATASET_NAME},{rel_path},{num_frames},{token_id_str}")

            if (i + 1) % 200 == 0:
                print(f"  [{subset}] {i + 1}/{len(ids)}")

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"[{subset}] wrote {len(rows)} rows -> {csv_path}")

    if unknown_tokens:
        from collections import Counter

        c = Counter(unknown_tokens)
        print(f"[info] {sum(c.values())} unmapped phoneme tokens "
              f"({len(c)} distinct), most common: {c.most_common(10)}")
    print(f"[done] preprocessed corpus at {args.out_dir}")
    print(f"       point data.dataset.root_dir at this directory.")


if __name__ == "__main__":
    main()
