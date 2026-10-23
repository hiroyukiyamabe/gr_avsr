#!/usr/bin/env python
"""Per-frame gate weight vs speech activity and segmental SNR (Phase 2c, strong form, Part A).

CLAUDE.md section 5.2 Part A. No GPU, no new eval: reads the per-frame audio
gate traces that eval_sweep_utt.py saved for the 8 fixed trace utterances
(decode.trace_utterances) at the 8 white-noise levels (exp/per_utt_sweep/), and
rebuilds each utterance's model input from (noise_seed = 0, utterance ID) with
the AddNoise / AudioTransform code path, checked against the logged audio_sha1.

Pre-registered (fixed in this file and committed before it was first run on the
traces; never change these after seeing results):
  - Frame t = clean audio samples [640 t, 640 (t + 1)) of the cut/padded model
    input (= video frame t = gate frame t).
  - VAD on the clean audio: E_t = mean square of frame t; speech iff
    10 log10(E_t) >= max_t 10 log10(E_t) - 30 dB (per utterance). Sensitivity:
    the same at -20 and -40 dB (reported, not used for conclusions).
  - Segmental SNR at each noisy level: 10 log10(sum speech^2 / sum noise^2)
    within frame t, noise = the scaled noise actually added (noisy - clean,
    before the instance layer norm). Energies floored at 1e-12.
  - Per utterance and level: mean gate weight in speech and in non-speech
    frames, and their difference; Spearman rho between per-frame weight and
    segmental SNR (noisy levels); at clean, Spearman rho with the frame speech
    energy (dB) instead, since segmental SNR is infinite.
  - "Converging stretches" (RESULTS.md 2.3 finding 3): per frame, the sd of
    the weight across the 8 levels; mean sd in speech vs non-speech frames and
    the AUC of the sd separating non-speech (low) from speech (high) frames.

Scope (CLAUDE.md 5.2): under stationary white noise, segmental SNR is driven by
speech energy, so it is confounded with speech activity. This can show whether
the converging stretches are pauses; it cannot separate reliability from
speech activity (Part B does that).

Outputs (new files only; refuses to overwrite):
  exp/per_utt_sweep/partA_summary.md, partA_summary.json, partA_frames.csv,
  partA_traces.png

Usage (from auto_avsr/):
  python analyze_gate_partA.py
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_per_utt_sweep import ASC, C_INK, C_INK2, C_REF, RAMP, auc, label, spearman, style
from datamodule.av_dataset import cut_or_pad, load_audio, load_video
from datamodule.transforms import AddNoise, AudioTransform

VAD_DB = 30.0
VAD_SENSITIVITY_DB = (20.0, 40.0)
ENERGY_FLOOR = 1e-12
RATE_RATIO = 640
NOISE_SEED = 0
CLEAN = 999999
EXPECTED_NOISE_SHA256 = "b031f778cc3b1f7b5617ae21718ec0e950006e77731407b6d8597163055a0e50"


def git_state():
    root = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=root).stdout.splitlines()
    return {"head": head, "dirty": bool(dirty), "dirty_paths": dirty}


def vad(energy_db, drop_db):
    return energy_db >= energy_db.max() - drop_db


def frame_energy(x, n):
    """x: 640 n samples -> per-frame sum of squares (float64)."""
    return (x.astype(np.float64).reshape(n, RATE_RATIO) ** 2).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="exp/per_utt_sweep")
    ap.add_argument("--static", default="rohan_avsr_3448_static")
    ap.add_argument("--gated", default="rohan_avsr_3448_gated")
    args = ap.parse_args()

    outs = {k: os.path.join(args.dir, f"partA_{k}") for k in ("summary.md", "summary.json", "frames.csv", "traces.png")}
    existing = [p for p in outs.values() if os.path.exists(p)]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing}; move them first")

    metas, rows, traces = {}, {}, {}
    for arm, exp in (("static", args.static), ("gated", args.gated)):
        for snr in ASC:
            stem = os.path.join(args.dir, exp, f"snr_{snr}")
            with open(stem + "_meta.json", encoding="utf-8") as f:
                metas[arm, snr] = json.load(f)
            with open(stem + ".csv", newline="", encoding="utf-8") as f:
                rows[arm, snr] = {r["utt_id"]: r for r in csv.DictReader(f)}
            with open(stem + "_traces.json", encoding="utf-8") as f:
                traces[arm, snr] = {u: np.array(t, dtype=np.float64) for u, t in json.load(f)["gate_w_audio"].items()}
    m0 = metas["gated", CLEAN]
    utts = m0["trace_utterances"]
    noise_path = metas["gated", -10]["noise_filename"]
    checks = []

    def check(ok, msg):
        if not ok:
            raise SystemExit(f"CHECK FAILURE: {msg}")
        checks.append(msg)

    check(all(m["noise_seed"] == NOISE_SEED for m in metas.values()), "all section 5.1 runs used noise_seed 0")
    check(all(m["noise_sha256"] == EXPECTED_NOISE_SHA256 for m in metas.values()), "all section 5.1 runs used NoiseX-92 white (white_16k.wav)")
    check(all(m["trace_utterances"] == utts for m in metas.values()) and len(utts) == 8, "8 fixed trace utterances in every run")
    check(all(np.all(traces["static", s][u] == 0.5) for s in ASC for u in utts), "static traces exactly 0.5 at every frame")

    root = m0["dataset_root"]
    with open(os.path.join(root, "labels", m0["test_file"]), encoding="utf-8") as f:
        label_rows = {os.path.splitext(os.path.basename(l.split(",")[1]))[0]: l.split(",") for l in f.read().splitlines()}

    per_utt, frame_rows, plot_data = [], [], {}
    for utt in utts:
        dataset_name, rel_path, n_label, _ = label_rows[utt]
        path = os.path.join(root, dataset_name, rel_path)
        n = len(load_video(path))
        raw = load_audio(path)
        # Frame alignment (CLAUDE.md 5.2): gate trace, video frames, audio / 640.
        lengths = {len(traces[a, s][utt]) for a in ("static", "gated") for s in ASC}
        offset = raw.shape[0] - RATE_RATIO * n
        check(lengths == {n} and int(n_label) == n and abs(offset) < RATE_RATIO,
              f"{utt}: trace length = {n} video frames = label; raw audio {raw.shape[0]} samples = 640 x {n} {offset:+d}")
        clean = cut_or_pad(raw, n * RATE_RATIO)
        clean_np = clean[:, 0].numpy()
        e_speech = frame_energy(clean_np, n)
        e_db = 10 * np.log10(np.maximum(e_speech / RATE_RATIO, ENERGY_FLOOR))
        speech = vad(e_db, VAD_DB)
        sens = {d: vad(e_db, d) for d in VAD_SENSITIVITY_DB}
        W = np.stack([traces["gated", s][utt] for s in ASC])  # [8 levels, n]
        spread = W.std(axis=0)
        seg = {}
        for snr in ASC:
            # Rebuild the model input exactly as the eval did; check against the logged hash.
            model_in = AudioTransform("test", snr_target=snr, noise_filename=noise_path, noise_seed=NOISE_SEED)(clean, key=utt)
            sha1 = hashlib.sha1(model_in.contiguous().numpy().tobytes()).hexdigest()
            for arm in ("static", "gated"):
                check(sha1 == rows[arm, snr][utt]["audio_sha1"], f"{utt} {label(snr)} {arm}: rebuilt input matches logged audio_sha1")
            if snr != CLEAN:
                noisy = AddNoise(noise_filename=noise_path, snr_target=snr, seed=NOISE_SEED)(clean, key=utt)
                e_noise = frame_energy((noisy - clean)[:, 0].numpy(), n)
                seg[snr] = 10 * np.log10(np.maximum(e_speech, ENERGY_FLOOR) / np.maximum(e_noise, ENERGY_FLOOR))
        for k, snr in enumerate(ASC):
            w = W[k]
            ref = seg[snr] if snr != CLEAN else e_db
            rec = {
                "utt_id": utt, "snr_target": snr, "n_frames": n,
                "n_speech": int(speech.sum()), "n_nonspeech": int((~speech).sum()),
                "w_speech": float(w[speech].mean()), "w_nonspeech": float(w[~speech].mean()) if (~speech).any() else float("nan"),
                "spearman_vs": "segmental SNR" if snr != CLEAN else "frame speech energy (dB)",
                "spearman": spearman(w, ref),
            }
            rec["w_speech_minus_nonspeech"] = rec["w_speech"] - rec["w_nonspeech"]
            for d, s in sens.items():
                rec[f"w_speech_minus_nonspeech_vad{int(d)}"] = (float(w[s].mean() - w[~s].mean()) if (~s).any() and s.any() else float("nan"))
            per_utt.append(rec)
            for t in range(n):
                frame_rows.append({"utt_id": utt, "snr_target": snr, "frame": t, "gate_w_audio": w[t],
                                   "speech": int(speech[t]), "speech_energy_db": e_db[t],
                                   "segmental_snr_db": seg[snr][t] if snr != CLEAN else ""})
        plot_data[utt] = (W, speech, e_db)
        per_utt.append({
            "utt_id": utt, "snr_target": "all", "n_frames": n,
            "spread_speech": float(spread[speech].mean()),
            "spread_nonspeech": float(spread[~speech].mean()) if (~speech).any() else float("nan"),
            "auc_spread_speech_vs_nonspeech": auc(spread[speech], spread[~speech]) if (~speech).any() else float("nan"),
        })

    with open(outs["frames.csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)
    summary = {"pre_registered": {"vad_drop_db": VAD_DB, "vad_sensitivity_db": list(VAD_SENSITIVITY_DB),
                                  "energy_floor": ENERGY_FLOOR, "noise_seed": NOISE_SEED},
               "checks": checks, "per_utt": per_utt, "git": git_state()}
    with open(outs["summary.json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_markdown(outs["summary.md"], checks, per_utt, utts)
    plot(plot_data, outs["traces.png"])
    print(open(outs["summary.md"], encoding="utf-8").read())


def write_markdown(path, checks, per_utt, utts):
    level = [r for r in per_utt if r["snr_target"] != "all"]
    spread = {r["utt_id"]: r for r in per_utt if r["snr_target"] == "all"}
    L = ["# Part A: per-frame gate weight vs speech activity and segmental SNR", "",
         "Generated by `analyze_gate_partA.py` (CLAUDE.md section 5.2 Part A; pre-registered settings in its docstring). "
         "Gated arm, 8 fixed trace utterances, seeded NoiseX-92 white (section 5.1 traces). "
         f"VAD: speech iff frame energy >= utterance max - {VAD_DB:.0f} dB (clean audio).", "",
         "## Checks (all passed)", ""]
    L += [f"- {c}" for c in checks]
    L += ["", "## Mean audio weight, speech minus non-speech frames", "",
          "| utterance | speech / non-speech frames | " + " | ".join(label(s) for s in ASC) + " |",
          "|---|---|" + "---:|" * len(ASC)]
    for u in utts:
        rs = {r["snr_target"]: r for r in level if r["utt_id"] == u}
        L.append(f"| {u} | {rs[CLEAN]['n_speech']} / {rs[CLEAN]['n_nonspeech']} | "
                 + " | ".join(f"{rs[s]['w_speech_minus_nonspeech']:+.3f}" for s in ASC) + " |")
    L += ["", "## Mean audio weight in non-speech frames / speech frames", "",
          "| utterance | " + " | ".join(label(s) for s in ASC) + " |", "|---|" + "---|" * len(ASC)]
    for u in utts:
        rs = {r["snr_target"]: r for r in level if r["utt_id"] == u}
        L.append(f"| {u} | " + " | ".join(f"{rs[s]['w_nonspeech']:.3f} / {rs[s]['w_speech']:.3f}" for s in ASC) + " |")
    L += ["", "## Spearman rho, per-frame weight vs segmental SNR (clean: vs frame speech energy)", "",
          "| utterance | " + " | ".join(label(s) for s in ASC) + " |", "|---|" + "---:|" * len(ASC)]
    for u in utts:
        rs = {r["snr_target"]: r for r in level if r["utt_id"] == u}
        L.append(f"| {u} | " + " | ".join(f"{rs[s]['spearman']:+.3f}" for s in ASC) + " |")
    for s in ASC:
        vals = [r["spearman"] for r in level if r["snr_target"] == s]
        diffs = [r["w_speech_minus_nonspeech"] for r in level if r["snr_target"] == s]
        L.append(f"- {label(s)}: median rho {np.median(vals):+.3f} (min {min(vals):+.3f}, max {max(vals):+.3f}); "
                 f"median speech - non-speech {np.median(diffs):+.3f}, positive in {sum(d > 0 for d in diffs)}/8")
    L += ["", "## Converging stretches: sd of the weight across the 8 levels, per frame", "",
          "| utterance | mean sd, speech | mean sd, non-speech | AUC (sd higher in speech) |", "|---|---:|---:|---:|"]
    for u in utts:
        r = spread[u]
        L.append(f"| {u} | {r['spread_speech']:.3f} | {r['spread_nonspeech']:.3f} | {r['auc_spread_speech_vs_nonspeech']:.3f} |")
    L += ["", "## VAD sensitivity (speech - non-speech weight, median over the 8 utterances)", "",
          "| VAD | " + " | ".join(label(s) for s in ASC) + " |", "|---|" + "---:|" * len(ASC)]
    for key, name in (("w_speech_minus_nonspeech_vad20", "-20 dB"), ("w_speech_minus_nonspeech", f"-{VAD_DB:.0f} dB (primary)"),
                      ("w_speech_minus_nonspeech_vad40", "-40 dB")):
        L.append(f"| {name} | " + " | ".join(f"{np.nanmedian([r[key] for r in level if r['snr_target'] == s]):+.3f}" for s in ASC) + " |")
    L += ["", "Scope: under stationary white noise, segmental SNR tracks speech energy, so it is confounded "
          "with speech activity. This cannot separate reliability from speech activity (Part B does).", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def plot(plot_data, out):
    fig, axes = plt.subplots(4, 2, figsize=(13, 9.5), sharey=True)
    for ax, (utt, (W, speech, e_db)) in zip(axes.ravel(), plot_data.items()):
        t = np.arange(W.shape[1]) / 25.0
        ax.fill_between(t, 0, 1, where=~speech, step="mid", color=C_REF, alpha=0.18, lw=0)
        for k, snr in enumerate(ASC):
            ax.plot(t, W[k], color=RAMP[k], lw=1.1, label=label(snr))
        ax.axhline(0.5, color=C_INK2, ls="--", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{utt} ({W.shape[1]} frames)", color=C_INK, fontsize=9, loc="left")
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("time (s); gray = non-speech (VAD, clean audio)", color=C_INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("audio gate weight", color=C_INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, fontsize=9,
               title="target SNR (gated arm)", title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
