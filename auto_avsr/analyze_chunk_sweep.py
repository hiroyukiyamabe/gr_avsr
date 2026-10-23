#!/usr/bin/env python
"""Analyse the chunk-wise corruption sweep written by eval_chunk_sweep.py.

Phase 2c, strong form, Part B (CLAUDE.md section 5.2). Written and committed
before any real run; every choice below is pre-registered and must not be
changed after results exist. Runs on the frozen model_avg_10 outputs only;
nothing here feeds back into training.

Runs read: AVSR static + gated x {audio_chunks, video_chunks, both_chunks,
placebo}; ASR x {audio_chunks, placebo}; VSR x {video_chunks, placebo}. (ASR
ignores video and VSR ignores audio, so ASR under both_chunks = ASR under
audio_chunks and VSR under both_chunks = VSR under video_chunks by construction.)

1. Integrity checks (abort on failure): complete and identical utterance sets;
   identical masks in every run; identical inputs across the twin arms; each
   condition corrupts exactly its streams; placebo inputs and CER/gate
   bit-identical to the clean per-utterance sweep (section 5.1); static arm
   exactly 0.5 at every frame; rows reproduce the logged aggregates; eval
   hparams differ only in the arm flag / condition; pre-registered settings,
   checkpoints, clean git tree at one HEAD.

2. Primary measures (gated arm), per utterance, paired within the utterance.
   Frames are split by the condition's masks into audio-only, video-only, both,
   neither (the placebo is read with each condition's masks as if applied):
     dA = mean audio weight in audio-only frames - mean in neither frames
          (expected < 0; condition audio_chunks)
     dV = mean audio weight in video-only frames - mean in neither frames
          (expected > 0; condition video_chunks)
   Reported for every condition, each beside the placebo under the same masks,
   twice: all frames (k = 0) and with k = 2 frames excluded on each side of every
   chunk boundary of either stream. Per measure: median over utterances with a
   95 % bootstrap CI (B = 10,000 utterance resamples, seed 0), fraction of
   utterances with the expected sign (a tie is not the expected sign),
   frame-level AUC pooled over all utterances' frames (oriented so that 0.5 =
   no separation and > 0.5 = the expected direction), and the median
   per-utterance AUC. Secondary: the placebo-corrected effect, per utterance
   d(condition) - d(placebo) (same masks and frames); the "both" frames in
   both_chunks; dV split by video corruption type; chunk-locked averages of
   w(condition) - w(placebo).

   Pre-registered verdict (k = 2 is the primary reading, k = 0 is shown beside):
     a measure HOLDS CLEARLY if its median's 95 % CI excludes 0 on the expected
       side and >= 75 % of utterances have the expected sign;
     its placebo is ~0 if the placebo median's 95 % CI includes 0 or
       |placebo median| <= 0.2 x |condition median|.
   PASS (per-frame weight tracks the corrupted stream, for these corruption
   types only) iff dA in audio_chunks and dV in video_chunks both hold clearly
   with their placebo ~0. Otherwise the gate is reported as reliability-
   sensitive at the utterance level only (section 5.1), naming which part
   failed. both_chunks is secondary and does not enter the verdict.

3. CER per condition: gated - static with a paired bootstrap 95 % CI (same
   resample at every condition); each AVSR arm against the better single
   modality for that condition (ASR: audio_chunks if audio is corrupted, else
   placebo; VSR: video_chunks if video is corrupted, else placebo), paired
   bootstrap. Gaps under ~0.5 pt are not called wins (CLAUDE.md section 8).

Usage (from auto_avsr/):
  python analyze_chunk_sweep.py --dir exp/chunk_sweep
"""
import argparse
import csv
import json
import os

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_per_utt_sweep import C_INK, C_INK2, C_REF, C_SERIES, flatten, style

CONDITIONS = ["audio_chunks", "video_chunks", "both_chunks", "placebo"]
CORRUPTED = {"audio_chunks": (True, False), "video_chunks": (False, True),
             "both_chunks": (True, True), "placebo": (False, False)}
# Masks each condition is read with (placebo: as both_chunks), as in eval_chunk_sweep.py.
READ_AS = {"audio_chunks": (True, False), "video_chunks": (False, True), "both_chunks": (True, True)}
MEASURES = {"dA": ("audio_only", -1), "dV": ("video_only", +1)}  # category vs neither, expected sign
PRIMARY = {"dA": "audio_chunks", "dV": "video_chunks"}
BOUNDARY_K = 2
KS = (0, BOUNDARY_K)
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 0
SIGN_FRAC_MIN = 0.75
PLACEBO_RATIO_MAX = 0.2
VIDEO_TYPE = {1: "occlusion", 2: "blur"}
# Pinned inputs (CLAUDE.md 5.2 and decode.chunk).
CKPT_PREFIX = {"static": "f543646f", "gated": "6f0291cb"}
WHITE_SHA256 = "b031f778cc3b1f7b5617ae21718ec0e950006e77731407b6d8597163055a0e50"
SPLIT_SHA256 = "e9415b3cd214a09463e1bbd7974e38fc7877cecae64ca605c91c079f1b9033d2"
ARM_KEYS = {"exp_name", "pretrained_model_path", "model.audiovisual_backbone.fusion_gate_freeze"}
CLEAN = 999999
C_PLACEBO = "#c9c8c3"


# ---------------------------------------------------------------- loading ---

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("n_frames", "edit_distance", "ref_len"):
            r[k] = int(r[k])
    return rows


def load_run(sweep_dir, exp, cond):
    d = os.path.join(sweep_dir, exp, cond)
    rows = read_rows(os.path.join(d, "rows.csv"))
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    z = np.load(os.path.join(d, "frames.npz"))
    off = z["offsets"]
    frames = {
        "utt_ids": [str(u) for u in z["utt_ids"]],
        "audio_mask": [z["audio_mask"][off[i]:off[i + 1]] for i in range(len(off) - 1)],
        "video_mask": [z["video_mask"][off[i]:off[i + 1]] for i in range(len(off) - 1)],
        "video_type": [z["video_type"][off[i]:off[i + 1]] for i in range(len(off) - 1)],
    }
    if "gate_w_audio" in z:
        frames["w"] = [z["gate_w_audio"][off[i]:off[i + 1]].astype(np.float64) for i in range(len(off) - 1)]
    with open(os.path.join(meta["eval_log_dir"], "hparams.yaml"), encoding="utf-8") as f:
        hp = flatten(yaml.safe_load(f))
    return {"rows": rows, "meta": meta, "frames": frames, "hparams": hp}


def load(args):
    runs = {}
    for arm, exp in (("static", args.static), ("gated", args.gated)):
        for cond in CONDITIONS:
            runs[arm, cond] = load_run(args.dir, exp, cond)
    for cond in ("audio_chunks", "placebo"):
        runs["asr", cond] = load_run(args.dir, args.asr, cond)
    for cond in ("video_chunks", "placebo"):
        runs["vsr", cond] = load_run(args.dir, args.vsr, cond)
    return runs


# -------------------------------------------------------------- integrity ---

def integrity(runs, args):
    report = []

    def check(ok, msg):
        if not ok:
            raise SystemExit(f"INTEGRITY FAILURE: {msg}")
        report.append(msg)

    ref = runs["static", "placebo"]["rows"]
    utts = [r["utt_id"] for r in ref]
    n = args.expect_n
    check(len(utts) == n and len(set(utts)) == n, f"{n} unique test utterances per run")
    heads = set()
    chunk_cfg = runs["static", "placebo"]["meta"]["chunk"]
    for (arm, cond), run in runs.items():
        rows, m, fr = run["rows"], run["meta"], run["frames"]
        tag = f"{arm} {cond}"
        check([r["utt_id"] for r in rows] == utts and fr["utt_ids"] == utts, f"{tag}: same utterances, same order (rows and frames)")
        check(all(r["ref_len"] == a["ref_len"] and r["n_frames"] == a["n_frames"] for r, a in zip(rows, ref)),
              f"{tag}: ref_len and n_frames match")
        check(m["condition"] == cond and m["smoke"] == args.smoke and m["n_utterances"] == n, f"{tag}: condition, smoke flag, utterance count")
        if not args.smoke:
            check(m["limit_test_batches"] is None, f"{tag}: full test set")
            check(m["git"]["dirty"] is False, f"{tag}: clean git tree")
        heads.add(m["git"]["head"])
        check(m["snr_target"] == CLEAN and m["noise_seed"] == 0, f"{tag}: clean outside the chunks, noise_seed 0")
        check(m["chunk"] == chunk_cfg and m["chunk"]["boundary_k"] == BOUNDARY_K, f"{tag}: pre-registered chunk settings (boundary k = {BOUNDARY_K})")
        check(m["chunk_noise_sha256"] == WHITE_SHA256 and m["occluder_split_sha256"] == SPLIT_SHA256,
              f"{tag}: NoiseX-92 white and the committed occluder split")
        check(m["max_abs_raw_audio_offset_samples"] is None or m["max_abs_raw_audio_offset_samples"] < 640,
              f"{tag}: raw audio within one frame of 640 x video frames")
        edits = sum(r["edit_distance"] for r in rows)
        chars = sum(r["ref_len"] for r in rows)
        check(abs(edits / chars - m["cer_logged"]) < 1e-6, f"{tag}: rows reproduce logged CER")
        if "w" in fr:
            check(all(len(w) == r["n_frames"] for w, r in zip(fr["w"], rows)), f"{tag}: gate trace length = n_frames")
            gate = sum(w.sum() for w in fr["w"]) / sum(r["n_frames"] for r in rows)
            check(abs(gate - m["gate_w_audio_logged"]) < 1e-6, f"{tag}: frames reproduce logged gate_w_audio")
    check(len(heads) == 1, f"all runs at one git HEAD ({sorted(heads)})")

    # Masks: one draw per (seed, utterance, stream), whatever the run.
    base = runs["static", "placebo"]["rows"]
    for key, run in runs.items():
        check(all(r["mask_sha1"] == b["mask_sha1"] for r, b in zip(run["rows"], base)), f"{key[0]} {key[1]}: identical masks")
    am = runs["static", "placebo"]["frames"]["audio_mask"]
    vm = runs["static", "placebo"]["frames"]["video_mask"]
    check(all(a.any() for a in am) and all(v.any() for v in vm), "every utterance has >= 1 chunk per stream")

    # Checkpoints.
    for arm in ("static", "gated"):
        shas = {runs[arm, c]["meta"]["checkpoint_sha256"] for c in CONDITIONS}
        check(len(shas) == 1 and shas.pop().startswith(CKPT_PREFIX[arm]), f"{arm}: section 5.1 checkpoint {CKPT_PREFIX[arm]}... in every condition")
    for arm, conds in (("asr", ("audio_chunks", "placebo")), ("vsr", ("video_chunks", "placebo"))):
        check(len({runs[arm, c]["meta"]["checkpoint_sha256"] for c in conds}) == 1, f"{arm}: one checkpoint")
    for cond in CONDITIONS:
        check(runs["static", cond]["meta"]["fusion_gate_freeze"] is True and runs["gated", cond]["meta"]["fusion_gate_freeze"] is False,
              f"{cond}: static frozen, gated trainable")

    # Static arm: exactly 0.5 at every frame.
    for cond in CONDITIONS:
        check(all(np.all(w == 0.5) for w in runs["static", cond]["frames"]["w"]), f"static {cond}: exactly 0.5 at every frame")

    # Inputs.
    def col(arm, cond, k):
        return [r[k] for r in runs[arm, cond]["rows"]]
    for cond in CONDITIONS:
        for k in ("audio_sha1", "video_sha1"):
            check(col("static", cond, k) == col("gated", cond, k), f"{cond}: twin arms saw bit-identical {k[:5]} inputs")
    same = lambda a, b: all(x == y for x, y in zip(a, b))
    differ = lambda a, b: all(x != y for x, y in zip(a, b))
    check(same(col("static", "video_chunks", "audio_sha1"), col("static", "placebo", "audio_sha1")), "video_chunks: audio clean")
    check(same(col("static", "audio_chunks", "video_sha1"), col("static", "placebo", "video_sha1")), "audio_chunks: video clean")
    check(differ(col("static", "audio_chunks", "audio_sha1"), col("static", "placebo", "audio_sha1")), "audio_chunks: audio corrupted in every utterance")
    check(differ(col("static", "video_chunks", "video_sha1"), col("static", "placebo", "video_sha1")), "video_chunks: video corrupted in every utterance")
    check(same(col("static", "both_chunks", "audio_sha1"), col("static", "audio_chunks", "audio_sha1")), "both_chunks: audio = audio_chunks audio")
    check(same(col("static", "both_chunks", "video_sha1"), col("static", "video_chunks", "video_sha1")), "both_chunks: video = video_chunks video")
    check(same(col("vsr", "video_chunks", "video_sha1"), col("static", "video_chunks", "video_sha1")), "VSR video_chunks input = AVSR video_chunks video")
    check(same(col("vsr", "placebo", "video_sha1"), col("static", "placebo", "video_sha1")), "VSR placebo input = AVSR placebo video")
    check(differ(col("asr", "audio_chunks", "audio_sha1"), col("asr", "placebo", "audio_sha1")), "ASR audio_chunks: audio corrupted in every utterance")

    # Placebo = the clean per-utterance sweep (section 5.1): same inputs, edits, CER and gate.
    for arm, exp in (("static", args.static), ("gated", args.gated)):
        stem = os.path.join(args.per_utt_dir, exp, f"snr_{CLEAN}")
        prev = read_rows(stem + ".csv")[:n]
        with open(stem + "_meta.json", encoding="utf-8") as f:
            prev_meta = json.load(f)
        rows = runs[arm, "placebo"]["rows"]
        check(all(r["audio_sha1"] == p["audio_sha1"] and r["edit_distance"] == p["edit_distance"] for r, p in zip(rows, prev)),
              f"{arm} placebo: audio input and edits per utterance identical to the clean section 5.1 run")
        if not args.smoke:
            m = runs[arm, "placebo"]["meta"]
            check(m["cer_logged"] == prev_meta["cer_logged"] and m["gate_w_audio_logged"] == prev_meta["gate_w_audio_logged"],
                  f"{arm} placebo: CER {m['cer_logged']!r} and gate {m['gate_w_audio_logged']!r} bit-identical to section 5.1 clean")
    if not args.smoke:
        for arm, exp in (("asr", args.asr), ("vsr", args.vsr)):
            with open(os.path.join(args.exp_dir, f"snr_sweep_{exp}.csv"), newline="") as f:
                clean = [r for r in csv.DictReader(f) if int(r["snr_target"]) == CLEAN][0]
            check(abs(runs[arm, "placebo"]["meta"]["cer_logged"] - float(clean["cer"])) < 1e-9,
                  f"{arm} placebo: CER equals the clean section 2.2 eval ({clean['cer']})")

    # Eval hparams: twins differ only in the arm flag; conditions only in chunk_condition.
    for cond in CONDITIONS:
        hs, hg = runs["static", cond]["hparams"], runs["gated", cond]["hparams"]
        diff = {k for k in set(hs) | set(hg) if hs.get(k) != hg.get(k)}
        check(diff == ARM_KEYS, f"{cond}: eval hparams of the twins differ only in {sorted(ARM_KEYS)}")
    for arm, conds in (("static", CONDITIONS), ("gated", CONDITIONS), ("asr", ("audio_chunks", "placebo")), ("vsr", ("video_chunks", "placebo"))):
        h0 = runs[arm, conds[-1]]["hparams"]
        for cond in conds[:-1]:
            h = runs[arm, cond]["hparams"]
            diff = {k for k in set(h0) | set(h) if h0.get(k) != h.get(k)}
            check(diff == {"decode.chunk_condition"}, f"{arm} {cond}: hparams differ from placebo only in chunk_condition")
    return report


# --------------------------------------------------------------- measures ---

def rankdata(a):
    """Average ranks (ties share the mean rank), 0-based; O(n log n)."""
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=np.intp)
    inv[sorter] = np.arange(len(a))
    s = a[sorter]
    obs = np.r_[True, s[1:] != s[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1) - 1


def auc(hi, lo):
    """P(hi > lo) + 0.5 P(hi == lo) over all pairs (Mann-Whitney U / n1 n2)."""
    hi, lo = np.asarray(hi, dtype=float), np.asarray(lo, dtype=float)
    if len(hi) == 0 or len(lo) == 0:
        return float("nan")
    r = rankdata(np.concatenate([hi, lo]))
    u = r[: len(hi)].sum() + len(hi) - len(hi) * (len(hi) + 1) / 2
    return float(u / (len(hi) * len(lo)))


def boundary_exclusion(masks, n, k):
    """True for frames kept: not within k frames of any chunk boundary of ``masks``."""
    keep = np.ones(n, dtype=bool)
    if k == 0:
        return keep
    for m in masks:
        for b in np.nonzero(m[1:] != m[:-1])[0] + 1:  # boundary between frames b-1 and b
            keep[max(b - k, 0):b + k] = False
    return keep


def frame_sets(audio_mask, video_mask, read_as, k):
    a = audio_mask & read_as[0]
    v = video_mask & read_as[1]
    keep = boundary_exclusion([m for m, on in ((audio_mask, read_as[0]), (video_mask, read_as[1])) if on], len(a), k)
    return {"audio_only": a & ~v & keep, "video_only": v & ~a & keep, "both": a & v & keep, "neither": ~a & ~v & keep}


def per_utt(run, placebo, read_as, k, cat, extra=None):
    """Per-utterance d = mean w[cat] - mean w[neither] for run and placebo, plus the
    pooled frames for the AUC. ``extra``: optional per-utterance frame filter."""
    fr, fp = run["frames"], placebo["frames"]
    d_run, d_pl, keep_idx = [], [], []
    pooled = {"run": ([], []), "placebo": ([], [])}
    utt_auc = {"run": [], "placebo": []}
    for i in range(len(fr["w"])):
        sets = frame_sets(fp["audio_mask"][i], fp["video_mask"][i], read_as, k)
        sel = sets[cat] & (extra[i] if extra is not None else True)
        ref = sets["neither"]
        if not sel.any() or not ref.any():
            continue
        keep_idx.append(i)
        for tag, w in (("run", fr["w"][i]), ("placebo", fp["w"][i])):
            (d_run if tag == "run" else d_pl).append(w[sel].mean() - w[ref].mean())
            pooled[tag][0].append(w[sel])
            pooled[tag][1].append(w[ref])
            utt_auc[tag].append((w[sel], w[ref]))
    return np.array(d_run), np.array(d_pl), keep_idx, pooled, utt_auc


def boot_median_ci(x):
    """95 % percentile CI of the median over utterance resamples. The resample
    depends only on len(x) (fixed seed), so a measure and its placebo, defined
    on the same utterances, share it."""
    if len(x) == 0:
        return float("nan"), float("nan")
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(x), size=(BOOTSTRAP_B, len(x)))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(d, sign):
    lo, hi = boot_median_ci(d)
    return {
        "n_utts": int(len(d)),
        "median": float(np.median(d)) if len(d) else float("nan"),
        "ci_lo": lo, "ci_hi": hi,
        "frac_expected_sign": float((np.sign(d) == sign).mean()) if len(d) else float("nan"),
    }


def oriented_auc(sel, ref, sign):
    """> 0.5 = the expected direction (sel below ref for sign -1, above for +1)."""
    return auc(ref, sel) if sign < 0 else auc(sel, ref)


def measure_block(run, placebo, cond, name, k, extra=None):
    cat, sign = MEASURES[name]
    d_run, d_pl, _, pooled, utt_auc = per_utt(run, placebo, READ_AS[cond], k, cat, extra)
    out = {"condition": cond, "measure": name, "k": k, "expected_sign": sign}
    for tag, d in (("run", d_run), ("placebo", d_pl)):
        s = summarise(d, sign)
        if len(d):
            sel = np.concatenate(pooled[tag][0])
            ref = np.concatenate(pooled[tag][1])
            s["auc_pooled"] = oriented_auc(sel, ref, sign)
            s["auc_utt_median"] = float(np.median([oriented_auc(a, b, sign) for a, b in utt_auc[tag]]))
        else:
            s["auc_pooled"] = s["auc_utt_median"] = float("nan")
        out[tag] = s
    out["corrected"] = summarise(d_run - d_pl, sign)
    return out


def verdict_parts(block):
    r, p, sign = block["run"], block["placebo"], block["expected_sign"]
    ci_excludes = (r["ci_hi"] < 0) if sign < 0 else (r["ci_lo"] > 0)
    holds = bool(ci_excludes and r["frac_expected_sign"] >= SIGN_FRAC_MIN)
    placebo_ok = bool((p["ci_lo"] <= 0 <= p["ci_hi"]) or abs(p["median"]) <= PLACEBO_RATIO_MAX * abs(r["median"]))
    return {"holds_clearly": holds, "ci_excludes_0_expected_side": bool(ci_excludes), "placebo_ok": placebo_ok}


def gate_analysis(runs):
    pl = runs["gated", "placebo"]
    blocks = []
    for cond in ("audio_chunks", "video_chunks", "both_chunks"):
        for name in MEASURES:
            cat = MEASURES[name][0]
            if (cat == "audio_only" and not READ_AS[cond][0]) or (cat == "video_only" and not READ_AS[cond][1]):
                continue
            for k in KS:
                blocks.append(measure_block(runs["gated", cond], pl, cond, name, k))
    # "both" frames in both_chunks (descriptive): weight in both-corrupted frames minus neither.
    both = []
    for k in KS:
        d_run, d_pl, *_ = per_utt(runs["gated", "both_chunks"], pl, READ_AS["both_chunks"], k, "both")
        both.append({"k": k, "run": summarise(d_run, 0), "placebo": summarise(d_pl, 0)})
    # dV by video corruption type (descriptive).
    by_type = []
    vt = pl["frames"]["video_type"]
    for code, tname in VIDEO_TYPE.items():
        for k in KS:
            b = measure_block(runs["gated", "video_chunks"], pl, "video_chunks", "dV", k,
                              extra=[t == code for t in vt])
            b["video_type"] = tname
            by_type.append(b)
    primary = {}
    for name, cond in PRIMARY.items():
        blk = [b for b in blocks if b["measure"] == name and b["condition"] == cond and b["k"] == BOUNDARY_K][0]
        primary[name] = {"condition": cond, **verdict_parts(blk)}
    passed = all(p["holds_clearly"] and p["placebo_ok"] for p in primary.values())
    return {"blocks": blocks, "both_frames": both, "dV_by_type": by_type, "primary": primary, "pass": passed}


def chunk_locked(runs, cond, stream, pre=10, post=10, span=25):
    """Mean w(cond) - w(placebo) aligned to chunk onsets and offsets of ``stream``,
    over chunks of that stream in ``cond`` (gated arm). Overlapping chunks of one
    stream are merged (mask runs)."""
    fr, fp = runs["gated", cond]["frames"], runs["gated", "placebo"]["frames"]
    on = {t: [] for t in range(-pre, span)}
    off = {t: [] for t in range(-span, post)}
    for i in range(len(fr["w"])):
        delta = fr["w"][i] - fp["w"][i]
        m = fp[f"{stream}_mask"][i]
        n = len(m)
        edges = np.diff(np.r_[0, m.astype(np.int8), 0])
        starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        for s, e in zip(starts, ends):
            for t in on:
                if 0 <= s + t < n and (t < 0 or s + t < e):
                    on[t].append(delta[s + t])
            for t in off:
                if 0 <= e + t < n and (t >= 0 or e + t >= s):
                    off[t].append(delta[e + t])
    return ({t: float(np.mean(v)) if v else float("nan") for t, v in on.items()},
            {t: float(np.mean(v)) if v else float("nan") for t, v in off.items()})


# -------------------------------------------------------------------- CER ---

def cer_analysis(runs, n):
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    idx = rng.integers(0, n, size=(BOOTSTRAP_B, n))
    nl = np.array([r["ref_len"] for r in runs["static", "placebo"]["rows"]])

    def edits(arm, cond):
        return np.array([r["edit_distance"] for r in runs[arm, cond]["rows"]])

    def gap(ea, eb):
        boot = (ea[idx].sum(1) - eb[idx].sum(1)) / nl[idx].sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"delta": 100 * (ea.sum() - eb.sum()) / nl.sum(), "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                "better": int((ea < eb).sum()), "worse": int((ea > eb).sum()), "tie": int((ea == eb).sum())}

    out = []
    for cond in CONDITIONS:
        a_on, v_on = CORRUPTED[cond]
        es, eg = edits("static", cond), edits("gated", cond)
        ea = edits("asr", "audio_chunks" if a_on else "placebo")
        ev = edits("vsr", "video_chunks" if v_on else "placebo")
        best, eb = ("ASR", ea) if ea.sum() <= ev.sum() else ("VSR", ev)
        out.append({
            "condition": cond,
            "cer_static": 100 * es.sum() / nl.sum(), "cer_gated": 100 * eg.sum() / nl.sum(),
            "cer_asr": 100 * ea.sum() / nl.sum(), "cer_vsr": 100 * ev.sum() / nl.sum(),
            "asr_run": "audio_chunks" if a_on else "placebo", "vsr_run": "video_chunks" if v_on else "placebo",
            "gated_minus_static": gap(eg, es),
            "best_single": best,
            "static_minus_best": gap(es, eb), "gated_minus_best": gap(eg, eb),
        })
    return out


# ------------------------------------------------------------------ plots ---

def plot_measures(runs, gate, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, name in zip(axes, ("dA", "dV")):
        cond = PRIMARY[name]
        cat, sign = MEASURES[name]
        data, labels = [], []
        for k in KS:
            d_run, d_pl, *_ = per_utt(runs["gated", cond], runs["gated", "placebo"], READ_AS[cond], k, cat)
            data += [d_run, d_pl]
            labels += [f"{cond}\nk={k}", f"placebo\nk={k}"]
        pos = np.arange(len(data))
        rng = np.random.default_rng(0)
        for p, d in zip(pos, data):
            color = C_SERIES if p % 2 == 0 else C_PLACEBO
            ax.scatter(p + rng.uniform(-0.18, 0.18, len(d)), d, s=5, color=color, alpha=0.35, lw=0)
            ax.plot([p - 0.28, p + 0.28], [np.median(d)] * 2, color=C_INK, lw=2)
        ax.axhline(0, color=C_INK2, lw=1)
        ax.set_xticks(pos, labels, fontsize=8)
        want = "below 0" if sign < 0 else "above 0"
        ax.set_title(f"{name}: audio weight in {cat.replace('_', '-')} minus neither frames\n(gated arm; expected {want}; bar = median)",
                     color=C_INK, fontsize=9, loc="left")
        ax.set_ylabel("per-utterance difference in audio weight", color=C_INK2)
        style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_locked(locked, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, side in zip(axes, ("onset", "offset")):
        for (cond, stream), color in ((("audio_chunks", "audio"), C_SERIES), (("video_chunks", "video"), "#d6662a")):
            on, off = locked[cond]
            series = on if side == "onset" else off
            t = sorted(series)
            ax.plot(t, [series[x] for x in t], "o-", ms=3, color=color, lw=1.5, label=f"{stream} chunks ({cond})")
        ax.axhline(0, color=C_INK2, lw=1)
        ax.axvline(0 if side == "onset" else 0, color=C_REF, ls="--", lw=1)
        ax.set_xlabel(f"frames from chunk {side} (40 ms)", color=C_INK2)
        ax.set_title(f"aligned to chunk {side}", color=C_INK, fontsize=9, loc="left")
        style(ax)
    axes[0].set_ylabel("mean w(condition) - w(placebo)", color=C_INK2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_traces(runs, trace_ids, out):
    utts = runs["gated", "placebo"]["frames"]["utt_ids"]
    ids = [u for u in trace_ids if u in utts]
    if not ids:
        return
    fig, axes = plt.subplots(len(ids), 1, figsize=(11, 1.9 * len(ids)), squeeze=False)
    for ax, utt in zip(axes[:, 0], ids):
        i = utts.index(utt)
        fp = runs["gated", "placebo"]["frames"]
        t = np.arange(len(fp["w"][i])) / 25.0
        for m, color in ((fp["audio_mask"][i], C_SERIES), (fp["video_mask"][i], "#d6662a")):
            ax.fill_between(t, 0, 1, where=m, step="mid", color=color, alpha=0.12, lw=0)
        ax.plot(t, fp["w"][i], color=C_REF, lw=1.2, label="placebo (clean)")
        ax.plot(t, runs["gated", "audio_chunks"]["frames"]["w"][i], color=C_SERIES, lw=1.2, label="audio_chunks")
        ax.plot(t, runs["gated", "video_chunks"]["frames"]["w"][i], color="#d6662a", lw=1.2, label="video_chunks")
        ax.axhline(0.5, color=C_INK2, ls="--", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_title(utt, color=C_INK, fontsize=8, loc="left")
        style(ax)
    axes[0, 0].legend(ncol=3, frameon=False, fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("time (s); shading: blue = audio chunks, orange = video chunks", color=C_INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------- markdown ---

def fmt_block(s):
    return (f"{s['median']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] | {100 * s['frac_expected_sign']:.0f} % | "
            f"{s['auc_pooled']:.3f} / {s['auc_utt_median']:.3f} | {s['n_utts']}")


def write_markdown(path, checks, gate, cer, locked):
    L = ["# Chunk-wise corruption sweep: summary", "",
         "Generated by `analyze_chunk_sweep.py` (pre-registered; CLAUDE.md section 5.2 Part B). "
         "Gated arm unless stated; w = per-frame audio gate weight.", "",
         "## Integrity checks (all passed)", ""]
    L += [f"- {c}" for c in checks]
    L += ["", "## Verdict (pre-registered rule, k = %d)" % BOUNDARY_K, ""]
    for name, p in gate["primary"].items():
        L.append(f"- {name} in {p['condition']}: holds clearly = {p['holds_clearly']} "
                 f"(CI excludes 0 on the expected side = {p['ci_excludes_0_expected_side']}), placebo ~0 = {p['placebo_ok']}")
    L.append(f"- **{'PASS' if gate['pass'] else 'FAIL'}**: " + (
        "the per-frame weight tracks the corrupted stream, for these corruption types only."
        if gate["pass"] else "report the gate as reliability-sensitive at the utterance level only (section 5.1)."))
    L += ["", "## Primary measures", "",
          "d = mean w in the category frames minus mean w in neither frames, per utterance. "
          f"Median [95 % bootstrap CI, B = {BOOTSTRAP_B}] | share of utterances with the expected sign | "
          "frame AUC pooled / median per utterance (> 0.5 = expected direction) | utterances.", "",
          "| condition | measure | k | condition run | placebo (same masks) | placebo-corrected median [CI] | share |",
          "|---|---|---:|---|---|---|---:|"]
    for b in gate["blocks"]:
        c = b["corrected"]
        L.append(f"| {b['condition']} | {b['measure']} ({'<' if b['expected_sign'] < 0 else '>'} 0) | {b['k']} | "
                 f"{fmt_block(b['run'])} | {fmt_block(b['placebo'])} | "
                 f"{c['median']:+.4f} [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | {100 * c['frac_expected_sign']:.0f} % |")
    L += ["", "### Both-corrupted frames (both_chunks; descriptive)", "",
          "| k | both - neither, median [CI] | placebo |", "|---:|---|---|"]
    for b in gate["both_frames"]:
        r, p = b["run"], b["placebo"]
        L.append(f"| {b['k']} | {r['median']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] (n {r['n_utts']}) | "
                 f"{p['median']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] |")
    L += ["", "### dV by video corruption type (video_chunks; descriptive)", "",
          "| type | k | condition run | placebo |", "|---|---:|---|---|"]
    for b in gate["dV_by_type"]:
        L.append(f"| {b['video_type']} | {b['k']} | {fmt_block(b['run'])} | {fmt_block(b['placebo'])} |")
    L += ["", "### Chunk-locked mean w(condition) - w(placebo) (descriptive)", ""]
    for cond, (on, off) in locked.items():
        L.append(f"- {cond}, onset-aligned (t = -10..24): " + " ".join(f"{on[t]:+.3f}" for t in sorted(on)))
        L.append(f"- {cond}, offset-aligned (t = -25..9): " + " ".join(f"{off[t]:+.3f}" for t in sorted(off)))
    L += ["", "## CER (%)", "",
          "Delta in pt with paired bootstrap 95 % CI (utterances resampled, same draw for every row). "
          "ASR/VSR: the run of that model under the same corruption of its own stream.", "",
          "| condition | static | gated | gated - static [CI] | better / worse / tie | ASR | VSR | static - best single [CI] | gated - best single [CI] |",
          "|---|---:|---:|---|---|---:|---:|---|---|"]
    for r in cer:
        g, s, b = r["gated_minus_static"], r["static_minus_best"], r["gated_minus_best"]
        L.append(f"| {r['condition']} | {r['cer_static']:.2f} | {r['cer_gated']:.2f} | {g['delta']:+.2f} [{g['ci_lo']:+.2f}, {g['ci_hi']:+.2f}] | "
                 f"{g['better']} / {g['worse']} / {g['tie']} | {r['cer_asr']:.2f} ({r['asr_run']}) | {r['cer_vsr']:.2f} ({r['vsr_run']}) | "
                 f"{s['delta']:+.2f} [{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}] vs {r['best_single']} | "
                 f"{b['delta']:+.2f} [{b['ci_lo']:+.2f}, {b['ci_hi']:+.2f}] |")
    L += ["", "Gaps under ~0.5 pt are not called wins (CLAUDE.md section 8).", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="exp/chunk_sweep")
    ap.add_argument("--per-utt-dir", default="exp/per_utt_sweep")
    ap.add_argument("--exp-dir", default="exp")
    ap.add_argument("--static", default="rohan_avsr_3448_static")
    ap.add_argument("--gated", default="rohan_avsr_3448_gated")
    ap.add_argument("--asr", default="rohan_asr_3448")
    ap.add_argument("--vsr", default="rohan_vsr_3448")
    ap.add_argument("--expect-n", type=int, default=400)
    ap.add_argument("--smoke", action="store_true", help="smoke outputs (partial test set, dirty tree allowed)")
    ap.add_argument("--trace-utterances", nargs="*", default=[
        "ROHAN4600_2664", "ROHAN4600_2549", "ROHAN4600_4174", "ROHAN4600_4519",
        "ROHAN4600_2086", "ROHAN4600_3717", "ROHAN4600_3506", "ROHAN4600_3191"])  # decode.trace_utterances
    args = ap.parse_args()

    runs = load(args)
    checks = integrity(runs, args)
    gate = gate_analysis(runs)
    cer = cer_analysis(runs, args.expect_n)
    locked = {cond: chunk_locked(runs, cond, stream) for cond, stream in (("audio_chunks", "audio"), ("video_chunks", "video"))}

    plot_measures(runs, gate, os.path.join(args.dir, "gate_chunk_measures.png"))
    plot_locked(locked, os.path.join(args.dir, "gate_chunk_locked.png"))
    plot_traces(runs, args.trace_utterances if not args.smoke else runs["gated", "placebo"]["frames"]["utt_ids"],
                os.path.join(args.dir, "gate_chunk_traces.png"))
    write_markdown(os.path.join(args.dir, "summary.md"), checks, gate, cer, locked)
    with open(os.path.join(args.dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"checks": checks, "gate": gate, "cer": cer,
                   "chunk_locked": {c: {"onset": on, "offset": off} for c, (on, off) in locked.items()}},
                  f, indent=2, default=float)
    print(open(os.path.join(args.dir, "summary.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
