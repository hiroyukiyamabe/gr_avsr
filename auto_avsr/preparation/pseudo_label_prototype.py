#! /usr/bin/env python
# -*- coding: utf-8 -*-
r"""Pseudo-labelling PROTOTYPE: Japanese ASR -> Katakana -> ROHAN-format labels.

Inference only. Runs a Japanese ASR model on a fixed random sample of ROHAN
*train* utterances, converts the ASR text to Katakana, and compares it with the
manual ROHAN transcript, so pseudo-label quality can be measured where ground
truth exists. Nothing here trains a model or touches val/test.

Conversion route (chosen to match how the ROHAN references were built):

    ASR text --pyopenjtalk.g2p--> phonemes --rohan_phonemes.phonemes_to_katakana--> Katakana

The ROHAN references come from the corpus's HTK phoneme alignments through
``rohan_phonemes.phonemes_to_katakana`` (see ``preprocess_rohan.py``). Sending the
pseudo-labels through the SAME phoneme -> Katakana function removes most
spelling-convention differences (particles are written as pronounced, long
vowels as repeated vowels, no "ー"). Differences that remain are real: ASR
errors, OpenJTalk misreading kanji, and any places where the corpus aligner and
OpenJTalk choose different phonemes for the same word.

CLAUDE.md section 6 rules enforced here:
  * output goes to its own directory, never inside the ROHAN root;
  * only the train split is accepted (never val/test; test transcripts unused);
  * provenance (source, ASR model + version, conversion tool, filtering) is written;
  * the script refuses to overwrite an existing output directory.

Usage (PowerShell, from auto_avsr\, venv active):

    $env:PYTHONIOENCODING = "utf-8"
    python preparation\pseudo_label_prototype.py --asr whisper --limit 300 `
      --out-dir ..\datasets\pseudo_label_proto\whisper_large_v3_train300

Dependencies (not in the current venv):
    pip install openai-whisper pyopenjtalk-plus
    (pyopenjtalk-plus has prebuilt Windows wheels and imports as "pyopenjtalk";
     plain pyopenjtalk is source-only and needs the MSVC C++ build tools)
    (reazonspeech backend: see https://github.com/reazon-research/ReazonSpeech)

Outputs in --out-dir:
    examples.tsv          one row per utterance (ASR text, phonemes, Katakana, reference, CER)
    examples_slide.md     the first 5 sampled utterances, for the midterm slide
    summary.json          corpus CER, per-utterance stats, unmapped phonemes
    provenance.json       everything needed to reproduce / audit the labels
    labels/rohan_pl_train_transcript_lengths.csv
                          ROHAN-format CSV with PSEUDO token ids (format demo only;
                          media paths are relative to the source ROHAN root)
"""

import argparse
import csv
import datetime
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preparation.rohan_phonemes import phonemes_to_katakana  # noqa: E402

# Baked in on purpose (CLAUDE.md section 9: data identity is not a CLI override).
# The default rohan.yaml root is the BROKEN rohan_preprocessed corpus.
ROHAN_ROOT = "C:/Users/Hiroyuki/gr_avsr/datasets/rohan_preprocessed_fixed"
SPLIT = "train"
CSV_NAME = "rohan_train_transcript_lengths.csv"
JA_DICT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "spm", "ja_char", "ja_char_units.txt")
N_SLIDE_EXAMPLES = 5


# ----------------------------------------------------------------------------
# Text -> Katakana
# ----------------------------------------------------------------------------

def g2p_phonemes(text):
    """ASR text -> OpenJTalk phoneme list in the ROHAN phoneme convention.

    pyopenjtalk writes devoiced vowels in upper case (e.g. "d e s U") and pauses
    as "pau". The ROHAN converter expects lower-case vowels and treats "pau" as
    silence, so only the vowels need folding. "N" (moraic nasal) stays as is.
    """
    import pyopenjtalk

    out = []
    for p in pyopenjtalk.g2p(text).split():
        if p in ("A", "I", "U", "E", "O"):
            p = p.lower()
        out.append(p)
    return out


def text_to_katakana(text, unknown=None):
    phonemes = g2p_phonemes(text) if text.strip() else []
    return phonemes, phonemes_to_katakana(phonemes, unknown=unknown)


def tokenize(kana, hashmap):
    unk = hashmap["<unk>"]
    return [int(hashmap.get(c, unk)) for c in kana if not c.isspace()]


def edit_distance(a, b):
    try:
        import editdistance
        return editdistance.eval(a, b)
    except ImportError:  # small pure-python fallback
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


# ----------------------------------------------------------------------------
# ASR backends
# ----------------------------------------------------------------------------

class WhisperASR:
    def __init__(self, model_name, device):
        import whisper
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_name, device=self.device)
        self.options = dict(language="ja", task="transcribe", temperature=0.0,
                            beam_size=5, condition_on_previous_text=False,
                            fp16=self.device == "cuda")
        self.info = {"backend": "openai-whisper", "model": model_name,
                     "package_version": getattr(whisper, "__version__", "unknown"),
                     "decode_options": self.options, "device": self.device}

    def __call__(self, audio_16k):
        # Passing an array (not a path) avoids Whisper's ffmpeg dependency.
        return self.model.transcribe(audio_16k, **self.options)["text"].strip()


class ReazonSpeechASR:
    """reazonspeech-nemo-v2 via the reazonspeech.nemo.asr package (optional)."""

    def __init__(self, device):
        from reazonspeech.nemo.asr import load_model

        self.model = load_model(device=device) if device else load_model()
        try:
            import importlib.metadata as md
            version = md.version("reazonspeech-nemo-asr")
        except Exception:
            version = "unknown"
        self.info = {"backend": "reazonspeech.nemo.asr", "model": "reazonspeech-nemo-v2",
                     "package_version": version, "decode_options": "package defaults",
                     "device": device or "package default"}

    def __call__(self, audio_16k):
        from reazonspeech.nemo.asr import audio_from_numpy, transcribe

        return transcribe(self.model, audio_from_numpy(audio_16k, 16000)).text.strip()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def read_split(root):
    path = os.path.join(root, "labels", CSV_NAME)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ds, rel, n_frames, token_ids = line.split(",", 3)
            utt = os.path.splitext(os.path.basename(rel))[0]
            rows.append({"dataset": ds, "rel_path": rel, "n_frames": int(n_frames),
                         "utt_id": utt, "token_ids": [int(t) for t in token_ids.split()]})
    return path, rows


def load_audio_16k(root, row):
    wav = os.path.join(root, row["dataset"], os.path.splitext(row["rel_path"])[0] + ".wav")
    data, sr = sf.read(wav, dtype="float32", always_2d=True)
    if sr != 16000:
        raise ValueError(f"{wav}: expected 16 kHz, got {sr}")
    return wav, np.ascontiguousarray(data.mean(axis=1))


def decode_reference(row, id_to_char):
    """Manual reference = the token ids in the split CSV, i.e. exactly the target
    the models train on. (normalize_rohan_video.py copies video, wav and the label
    CSVs into rohan_preprocessed_fixed, but not rohan_text_seg/*.txt.)"""
    bad = [i for i in row["token_ids"] if i not in id_to_char or id_to_char[i] == "<unk>"]
    if bad:
        raise ValueError(f"{row['utt_id']}: reference has unknown/<unk> token ids {bad[:5]}")
    return "".join(id_to_char[i] for i in row["token_ids"])


def git_state(repo_dir):
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status)}


def pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", choices=["whisper", "reazonspeech"], default="whisper")
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--limit", type=int, default=300,
                    help="number of train utterances to sample (0 = all 3800)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = os.path.normpath(ROHAN_ROOT)
    out_dir = os.path.normpath(os.path.abspath(args.out_dir))
    if os.path.commonpath([root, out_dir]) == root:
        sys.exit(f"refusing: --out-dir is inside the ROHAN root ({root}); "
                 "pseudo-labels must live in their own directory (CLAUDE.md section 6)")
    if os.path.exists(out_dir):
        sys.exit(f"refusing to overwrite existing {out_dir}; move it first")

    csv_path, rows = read_split(root)
    rng = random.Random(args.seed)
    sample = rows if args.limit in (0, None) or args.limit >= len(rows) \
        else rng.sample(rows, args.limit)
    # Sampling order is fixed before any ASR output exists; the first
    # N_SLIDE_EXAMPLES of it become the slide examples (chosen before looking).
    print(f"[info] {len(sample)} / {len(rows)} {SPLIT} utterances from {csv_path}")

    units = open(JA_DICT, encoding="utf-8").read().splitlines()
    hashmap = {u.split()[0]: u.split()[-1] for u in units}
    id_to_char = {int(i): c for c, i in hashmap.items()}

    # Check every input BEFORE loading the ASR model or creating --out-dir, so a
    # data problem fails fast and leaves nothing behind.
    refs = {}
    for row in sample:
        wav = os.path.join(root, row["dataset"], os.path.splitext(row["rel_path"])[0] + ".wav")
        if not os.path.isfile(wav):
            sys.exit(f"missing audio: {wav}")
        refs[row["utt_id"]] = decode_reference(row, id_to_char)
    print(f"[info] inputs OK: {len(sample)} wavs found, references decoded from the split CSV")

    asr = WhisperASR(args.whisper_model, args.device) if args.asr == "whisper" \
        else ReazonSpeechASR(args.device)
    print(f"[info] ASR: {asr.info['backend']} {asr.info['model']} on {asr.info['device']}")

    os.makedirs(os.path.join(out_dir, "labels"))
    results, unknown, csv_rows = [], [], []
    for i, row in enumerate(sample, 1):
        wav, audio = load_audio_16k(root, row)
        ref = refs[row["utt_id"]]
        hyp_text = asr(audio)
        phonemes, hyp = text_to_katakana(hyp_text, unknown=unknown)
        ed = edit_distance(hyp, ref)
        results.append({"utt_id": row["utt_id"], "asr_text": hyp_text,
                        "phonemes": " ".join(phonemes), "pseudo_katakana": hyp,
                        "reference": ref, "edit_distance": ed, "ref_len": len(ref),
                        "cer": ed / max(len(ref), 1)})
        ids = tokenize(hyp, hashmap)
        if ids:
            csv_rows.append(f"{row['dataset']},{row['rel_path']},{row['n_frames']},"
                            + " ".join(map(str, ids)))
        if i % 25 == 0 or i == len(sample):
            tot = sum(r["edit_distance"] for r in results) / sum(r["ref_len"] for r in results)
            print(f"  {i}/{len(sample)}  running CER {100 * tot:.2f} %")

    # --- examples.tsv
    fields = ["utt_id", "asr_text", "phonemes", "pseudo_katakana", "reference",
              "edit_distance", "ref_len", "cer"]
    with open(os.path.join(out_dir, "examples.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(results)

    # --- slide examples (first N in sampling order, fixed before looking)
    with open(os.path.join(out_dir, "examples_slide.md"), "w", encoding="utf-8") as f:
        f.write(f"# Pseudo-label examples ({asr.info['model']}, ROHAN train)\n\n"
                "First utterances of the seeded sample, chosen before any output was seen.\n\n")
        for r in results[:N_SLIDE_EXAMPLES]:
            f.write(f"## {r['utt_id']}  (CER {100 * r['cer']:.1f} %)\n\n"
                    f"- ASR text: {r['asr_text']}\n"
                    f"- Pseudo-label: {r['pseudo_katakana']}\n"
                    f"- Manual label: {r['reference']}\n\n")

    # --- pseudo-label CSV (format demo)
    with open(os.path.join(out_dir, "labels", "rohan_pl_train_transcript_lengths.csv"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows) + ("\n" if csv_rows else ""))

    # --- summary
    cers = [r["cer"] for r in results]
    total_ed = sum(r["edit_distance"] for r in results)
    total_ref = sum(r["ref_len"] for r in results)
    unk = Counter(unknown)
    summary = {
        "n_utterances": len(results),
        "corpus_cer": total_ed / max(total_ref, 1),
        "per_utt_cer": {"median": statistics.median(cers) if cers else None,
                        "mean": statistics.fmean(cers) if cers else None,
                        "share_exact": sum(c == 0 for c in cers) / max(len(cers), 1),
                        "share_le_5pct": sum(c <= 0.05 for c in cers) / max(len(cers), 1)},
        "empty_pseudo_labels": sum(not r["pseudo_katakana"] for r in results),
        "unmapped_phonemes": dict(unk.most_common()),
        "note": "CER is a fraction (x100 for %). Train split only. Preliminary.",
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --- provenance (CLAUDE.md section 6)
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    provenance = {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "prototype / quality check; NOT a training set",
        "source": {"rohan_root": root, "split": SPLIT, "split_csv": csv_path,
                   "audio": "16 kHz wav next to each video in the ROHAN root",
                   "n_sampled": len(sample), "n_in_split": len(rows),
                   "sampling_seed": args.seed,
                   "utterance_ids": sorted(r["utt_id"] for r in sample)},
        "asr": asr.info,
        "katakana_conversion": {
            "tool": "pyopenjtalk.g2p -> rohan_phonemes.phonemes_to_katakana",
            # The fork pyopenjtalk-plus (prebuilt Windows wheels) imports as
            # "pyopenjtalk" but ships its own dictionary, so record which one ran.
            "pyopenjtalk_distribution": next(
                (f"{d} {pkg_version(d)}" for d in ("pyopenjtalk-plus", "pyopenjtalk")
                 if pkg_version(d) != "unknown"), "unknown"),
            "devoiced_vowels": "upper-case A/I/U/E/O folded to lower case",
            "vocabulary": JA_DICT},
        "filtering": "none (no confidence filtering, no length filtering)",
        "reference_source": "token ids in the split CSV, decoded with the vocabulary above",
        "reference_labels_used_for": "measurement only, never to generate or filter labels",
        "csv_media_paths": "relative to the source ROHAN root, not this directory",
        "git": git_state(repo_dir),
        "python": sys.version.split()[0],
        "command": " ".join(sys.argv),
    }
    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)

    print(f"\n[done] corpus CER {100 * summary['corpus_cer']:.2f} % over {len(results)} utts "
          f"(median per-utt {100 * summary['per_utt_cer']['median']:.2f} %)")
    if unk:
        print(f"[warn] unmapped phonemes: {unk.most_common(10)}")
    print(f"       outputs in {out_dir}")


if __name__ == "__main__":
    main()
