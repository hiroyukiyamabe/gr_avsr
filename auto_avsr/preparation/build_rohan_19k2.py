#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the 19.2 kHz AVSR dataset (Kondo & Tamura's paper-faithful audio rate).

Takes an existing preprocessed dataset (videos already normalised, e.g.
``rohan_preprocessed_fixed``) and produces a sibling dataset with:

  * videos + label CSVs copied through UNCHANGED (bit-identical video stream,
    so results stay commensurable with the 16 kHz runs), and
  * every ``.wav`` regenerated from the RAW 96 kHz corpus audio, decimated
    exactly x5 to 19,200 Hz (never resampled from the 16 kHz files, which
    would bandlimit to 8 kHz).

Use with ``data.dataset.rate_ratio=768`` (= 19200 / 25 fps).

Example::

    python preparation/build_rohan_19k2.py \
        --src ../datasets/rohan_preprocessed_fixed \
        --dst ../datasets/rohan_preprocessed_avsr19k2 \
        --raw-wav-dir ../datasets/ROHAN4600/ROHAN4600_zumndamon_normal_synchronized_wav
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preparation.utils import save2aud  # noqa: E402

TARGET_RATE = 19200


def resample_raw_wav(raw_path, target_rate):
    data, rate = sf.read(raw_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))  # channels x frames
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return torchaudio.functional.resample(waveform, rate, target_rate)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="existing preprocessed root (videos + labels reused as-is)")
    ap.add_argument("--dst", required=True, help="output root for the 19.2 kHz dataset")
    ap.add_argument("--raw-wav-dir", required=True,
                    help="raw 96 kHz corpus wav folder (audio source of truth)")
    args = ap.parse_args()

    # labels: copied verbatim (frame counts and tokens don't depend on audio rate)
    shutil.copytree(os.path.join(args.src, "labels"),
                    os.path.join(args.dst, "labels"), dirs_exist_ok=True)

    src_seg = os.path.join(args.src, "rohan", "rohan_video_seg")
    dst_seg = os.path.join(args.dst, "rohan", "rohan_video_seg")
    os.makedirs(dst_seg, exist_ok=True)

    mp4s = sorted(glob.glob(os.path.join(src_seg, "*.mp4")))
    print(f"[info] {len(mp4s)} utterances: copying video, resampling raw 96k -> {TARGET_RATE} Hz",
          flush=True)
    for i, mp4 in enumerate(mp4s):
        utt_id = os.path.splitext(os.path.basename(mp4))[0]
        shutil.copyfile(mp4, os.path.join(dst_seg, f"{utt_id}.mp4"))
        raw_wav = os.path.join(args.raw_wav_dir, f"{utt_id}.wav")
        waveform = resample_raw_wav(raw_wav, TARGET_RATE)
        save2aud(os.path.join(dst_seg, f"{utt_id}.wav"), waveform, TARGET_RATE)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(mp4s)}", flush=True)
    print(f"[done] 19.2 kHz dataset at {args.dst}")


if __name__ == "__main__":
    main()
