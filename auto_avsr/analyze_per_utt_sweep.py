#!/usr/bin/env python
"""Analyse the per-utterance SNR sweep written by eval_sweep_utt.py.

Phase 2c, weak form (CLAUDE.md section 5.1): utterance-level audio gate weight
against the target SNR, plus paired bootstrap of the twin CER gap. Runs on the
frozen model_avg_10 outputs only; nothing here feeds back into training.

1. Integrity checks (abort on failure): complete and identical utterance sets,
   identical noisy audio across arms (audio_sha1), static arm exactly 0.5 at
   every frame, rows sum to the logged aggregates, eval hparams.yaml of the two
   arms differ only in the arm flag.
2. CER per arm and SNR, gated - static with a paired bootstrap 95 % CI
   (utterances resampled with replacement, same draw for both arms).
3. Gate vs target SNR (gated arm): distribution of utterance-mean audio weight
   per level, per-utterance Spearman rho across the 8 levels, adjacent-level
   separation (paired fraction, unpaired AUC), share of variance explained by
   the level, within- vs between-utterance spread.
4. Per-frame traces of the fixed trace utterances.

Usage (from auto_avsr/):
  python analyze_per_utt_sweep.py --dir exp/per_utt_sweep
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

CLEAN = 999999
LEVELS = [CLEAN, 20, 15, 10, 5, 0, -5, -10]       # sweep ladder (CLAUDE.md section 4)
ASC = [-10, -5, 0, 5, 10, 15, 20, CLEAN]          # low -> high SNR
# Arm flag that may (and must) differ between the twin evals.
ARM_KEYS = {"exp_name", "pretrained_model_path", "model.audiovisual_backbone.fusion_gate_freeze"}
# Aggregates of the earlier unseeded sweep (exp/snr_sweep_*.csv). Clean involves
# no noise draw, so the seeded per-utterance run must reproduce them exactly.
PREV_CLEAN = {
    "static": {"cer": 0.031336478888988495, "gate": 0.5},
    "gated": {"cer": 0.03524336591362953, "gate": 0.5370424389839172},
}
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 0

# Palette (dataviz reference): blue for the single series, gray reference,
# ordinal blue ramp (steps 250..700) for the 8 SNR levels, low -> high SNR.
C_SERIES = "#2a78d6"
C_REF = "#8a8984"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#0d366b"]


def label(snr):
    return "clean" if snr == CLEAN else f"{snr} dB"


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("n_frames", "edit_distance", "ref_len"):
            r[k] = int(r[k])
        for k in ("gate_w_audio_mean", "gate_w_audio_std", "gate_w_audio_min", "gate_w_audio_max"):
            r[k] = float(r[k])
    return rows


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def rankdata(a):
    """Average ranks (ties share the mean rank), 0-based."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a))
    for v in np.unique(a):
        tie = a == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(hi, lo):
    """P(hi > lo) + 0.5 P(hi == lo) over all pairs (Mann-Whitney U / n1 n2)."""
    hi, lo = np.asarray(hi), np.asarray(lo)
    r = rankdata(np.concatenate([hi, lo]))
    u = r[: len(hi)].sum() + len(hi) - len(hi) * (len(hi) + 1) / 2  # ranks are 0-based
    return float(u / (len(hi) * len(lo)))


def load(sweep_dir, static, gated):
    data = {}
    for arm, exp in (("static", static), ("gated", gated)):
        for snr in LEVELS:
            stem = os.path.join(sweep_dir, exp, f"snr_{snr}")
            rows = read_rows(stem + ".csv")
            with open(stem + "_meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            with open(stem + "_traces.json", encoding="utf-8") as f:
                traces = json.load(f)["gate_w_audio"]
            with open(os.path.join(meta["eval_log_dir"], "hparams.yaml"), encoding="utf-8") as f:
                hp = flatten(yaml.safe_load(f))
            data[arm, snr] = {"rows": rows, "meta": meta, "traces": traces, "hparams": hp}
    return data


def integrity(data):
    """Return a list of check strings; raise on any failure."""
    report = []

    def check(ok, msg):
        if not ok:
            raise SystemExit(f"INTEGRITY FAILURE: {msg}")
        report.append(msg)

    ref = data["static", CLEAN]["rows"]
    utts = [r["utt_id"] for r in ref]
    check(len(utts) == 400 and len(set(utts)) == 400, "400 unique test utterances per run")
    for (arm, snr), d in data.items():
        rows = d["rows"]
        check([r["utt_id"] for r in rows] == utts, f"{arm} {label(snr)}: same utterances, same order")
        check(all(r["ref_len"] == a["ref_len"] and r["n_frames"] == a["n_frames"] for r, a in zip(rows, ref)),
              f"{arm} {label(snr)}: ref_len and n_frames match")
        m = d["meta"]
        check(m["limit_test_batches"] is None and m["n_utterances"] == 400, f"{arm} {label(snr)}: full test set")
        edits = sum(r["edit_distance"] for r in rows)
        chars = sum(r["ref_len"] for r in rows)
        frames = sum(r["n_frames"] for r in rows)
        gate = sum(r["gate_w_audio_mean"] * r["n_frames"] for r in rows) / frames
        check(abs(edits / chars - m["cer_logged"]) < 1e-6, f"{arm} {label(snr)}: rows reproduce logged CER")
        check(abs(gate - m["gate_w_audio_logged"]) < 1e-6, f"{arm} {label(snr)}: rows reproduce logged gate_w_audio")
    for snr in LEVELS:
        s, g = data["static", snr]["rows"], data["gated", snr]["rows"]
        check(all(a["audio_sha1"] == b["audio_sha1"] for a, b in zip(s, g)),
              f"{label(snr)}: both arms saw bit-identical audio for all 400 utterances")
        check(all(r["gate_w_audio_min"] == 0.5 and r["gate_w_audio_max"] == 0.5 for r in s),
              f"{label(snr)}: static arm reads exactly 0.5 at every frame")
        if snr != CLEAN:
            clean = data["static", CLEAN]["rows"]
            check(all(a["audio_sha1"] != c["audio_sha1"] for a, c in zip(s, clean)),
                  f"{label(snr)}: noise was applied (audio differs from clean)")
        hs, hg = data["static", snr]["hparams"], data["gated", snr]["hparams"]
        diff = {k for k in set(hs) | set(hg) if hs.get(k) != hg.get(k)}
        check(diff == ARM_KEYS, f"{label(snr)}: eval hparams differ only in {sorted(ARM_KEYS)}")
    for arm in ("static", "gated"):
        h0 = data[arm, CLEAN]["hparams"]
        for snr in LEVELS[1:]:
            h = data[arm, snr]["hparams"]
            diff = {k for k in set(h0) | set(h) if h0.get(k) != h.get(k)}
            check(diff == {"decode.snr_target"}, f"{arm} {label(snr)}: hparams differ from clean only in snr_target")
        metas = [data[arm, snr]["meta"] for snr in LEVELS]
        check(len({m["checkpoint_sha256"] for m in metas}) == 1, f"{arm}: one checkpoint across all levels")
    metas = [d["meta"] for d in data.values()]
    check(len({(m["noise_sha256"], m["noise_seed"], m["dataset_root"]) for m in metas}) == 1,
          "all runs: same noise file, seed and dataset")
    check(data["static", CLEAN]["meta"]["checkpoint_sha256"] != data["gated", CLEAN]["meta"]["checkpoint_sha256"],
          "the two arms use different checkpoints")
    for arm in ("static", "gated"):
        m = data[arm, CLEAN]["meta"]
        same = m["cer_logged"] == PREV_CLEAN[arm]["cer"] and m["gate_w_audio_logged"] == PREV_CLEAN[arm]["gate"]
        check(same, f"{arm} clean: CER {m['cer_logged']!r} and gate {m['gate_w_audio_logged']!r} "
                    "bit-identical to the earlier sweep (logging changes nothing)")
    return report


def cer_table(data):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = 400
    idx = rng.integers(0, n, size=(BOOTSTRAP_B, n))  # one draw, reused at every level
    out = []
    for snr in LEVELS:
        s, g = data["static", snr]["rows"], data["gated", snr]["rows"]
        es = np.array([r["edit_distance"] for r in s])
        eg = np.array([r["edit_distance"] for r in g])
        nl = np.array([r["ref_len"] for r in s])
        cs, cg = es.sum() / nl.sum(), eg.sum() / nl.sum()
        boot = (eg[idx].sum(1) - es[idx].sum(1)) / nl[idx].sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({
            "snr": snr, "cer_static": 100 * cs, "cer_gated": 100 * cg, "delta": 100 * (cg - cs),
            "ci_lo": 100 * lo, "ci_hi": 100 * hi, "p_boot_gated_worse": float((boot > 0).mean()),
            "utts_gated_better": int((eg < es).sum()), "utts_gated_worse": int((eg > es).sum()),
            "utts_tie": int((eg == es).sum()),
        })
    return out


def gate_analysis(data):
    W = np.array([[r["gate_w_audio_mean"] for r in data["gated", snr]["rows"]] for snr in ASC])  # [8, 400]
    S = np.array([[r["gate_w_audio_std"] for r in data["gated", snr]["rows"]] for snr in ASC])
    per_level = []
    for k, snr in enumerate(ASC):
        w = W[k]
        q = np.percentile(w, [5, 25, 50, 75, 95])
        per_level.append({
            "snr": snr, "mean": float(w.mean()), "sd_between_utts": float(w.std()),
            "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4],
            "median_within_utt_sd": float(np.median(S[k])),
        })
    level_rank = np.arange(len(ASC))
    rhos = np.array([spearman(level_rank, W[:, j]) for j in range(W.shape[1])])
    monotone = np.all(np.diff(W, axis=0) > 0, axis=0)
    adjacent = []
    for k in range(len(ASC) - 1):
        adjacent.append({
            "lo": ASC[k], "hi": ASC[k + 1],
            "paired_frac_hi_gt_lo": float((W[k + 1] > W[k]).mean()),
            "auc_unpaired": auc(W[k + 1], W[k]),
        })
    grand = W.mean()
    eta2 = float(((W.mean(1, keepdims=True) - grand) ** 2).sum() * W.shape[1] / ((W - grand) ** 2).sum())
    pooled = spearman(np.repeat(level_rank, W.shape[1]), W.ravel())
    return {
        "per_level": per_level,
        "per_utt_spearman": {
            "median": float(np.nanmedian(rhos)), "frac_positive": float((rhos > 0).mean()),
            "frac_rho_1": float((rhos == 1).mean()), "min": float(np.nanmin(rhos)),
        },
        "frac_strictly_monotone": float(monotone.mean()),
        "adjacent": adjacent,
        "auc_clean_vs_minus10": auc(W[-1], W[0]),
        "eta2_level": eta2,
        "pooled_spearman": pooled,
    }, W


def trace_summary(data):
    trace_ids = data["gated", CLEAN]["meta"]["trace_utterances"]
    out = {}
    for utt in trace_ids:
        out[utt] = {}
        for snr in ASC:
            for arm in ("static", "gated"):
                t = np.array(data[arm, snr]["traces"][utt])
                if arm == "static" and not np.all(t == 0.5):
                    raise SystemExit(f"INTEGRITY FAILURE: static trace {utt} {label(snr)} != 0.5")
            t = np.array(data["gated", snr]["traces"][utt])
            out[utt][snr] = {"n": len(t), "mean": float(t.mean()), "sd": float(t.std()),
                             "min": float(t.min()), "max": float(t.max())}
    return trace_ids, out


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_REF)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.grid(True, axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def plot_gate(W, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(ASC))
    for j in range(W.shape[1]):
        ax.plot(x, W[:, j], color=C_REF, lw=0.6, alpha=0.12)
    q25, q50, q75 = np.percentile(W, [25, 50, 75], axis=1)
    ax.fill_between(x, q25, q75, color=C_SERIES, alpha=0.18, lw=0)
    ax.plot(x, q50, "o-", color=C_SERIES, lw=2, ms=8, mec="white", mew=2)
    ax.axhline(0.5, color=C_INK2, ls="--", lw=1)
    ax.text(len(ASC) - 0.95, 0.5, "static arm\n(0.5, exact)", color=C_INK2, fontsize=8, va="center")
    ax.set_xticks(x, [label(s) for s in ASC])
    ax.set_xlim(-0.3, len(ASC) - 0.1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("target SNR (white noise, seeded)", color=C_INK2)
    ax.set_ylabel("utterance-mean audio gate weight", color=C_INK2)
    ax.set_title("Gated arm: median and IQR over 400 test utterances\n(gray: each utterance across the 8 levels)",
                 color=C_INK, fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_delta(table, out):
    rows = sorted(table, key=lambda r: ASC.index(r["snr"]))
    x = np.arange(len(rows))
    d = np.array([r["delta"] for r in rows])
    lo = d - np.array([r["ci_lo"] for r in rows])
    hi = np.array([r["ci_hi"] for r in rows]) - d
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.axhline(0, color=C_INK2, lw=1)
    ax.errorbar(x, d, yerr=[lo, hi], fmt="o", color=C_SERIES, ms=8, mec="white", mew=2,
                elinewidth=2, capsize=0)
    ax.set_xticks(x, [label(r["snr"]) for r in rows])
    ax.set_xlabel("target SNR (white noise, seeded)", color=C_INK2)
    ax.set_ylabel("CER gated - static (pt)", color=C_INK2)
    ax.set_title("Twin CER gap with paired-bootstrap 95% CI (above 0 = static better)",
                 color=C_INK, fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_traces(data, trace_ids, out):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True)
    for ax, utt in zip(axes.ravel(), trace_ids):
        for k, snr in enumerate(ASC):
            t = data["gated", snr]["traces"][utt]
            ax.plot(np.arange(len(t)) / 25.0, t, color=RAMP[k], lw=1.2, label=label(snr))
        ax.axhline(0.5, color=C_INK2, ls="--", lw=0.8)
        ax.set_title(f"{utt} ({len(t)} frames)", color=C_INK, fontsize=9, loc="left")
        ax.set_ylim(0, 1)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("time (s)", color=C_INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("audio gate weight", color=C_INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, fontsize=9,
               title="target SNR (gated arm; dashed = static arm, 0.5)", title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_markdown(path, checks, table, gate, traces, trace_ids):
    L = ["# Per-utterance SNR sweep: summary", "",
         "Generated by `analyze_per_utt_sweep.py`. CER in %, Delta in pt (gated - static).", "",
         "## Integrity checks (all passed)", ""]
    L += [f"- {c}" for c in checks]
    L += ["", "## CER and paired bootstrap", "",
          f"B = {BOOTSTRAP_B}, utterances resampled with replacement (seed {BOOTSTRAP_SEED}).", "",
          "| SNR | static | gated | Delta | 95% CI | P(boot Delta > 0) | utts gated better / worse / tie |",
          "|---|---:|---:|---:|---|---:|---|"]
    for r in table:
        L.append(f"| {label(r['snr'])} | {r['cer_static']:.2f} | {r['cer_gated']:.2f} | {r['delta']:+.2f} | "
                 f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_boot_gated_worse']:.3f} | "
                 f"{r['utts_gated_better']} / {r['utts_gated_worse']} / {r['utts_tie']} |")
    L += ["", "## Gate vs target SNR (gated arm, utterance-mean audio weight)", "",
          "| SNR | mean | sd between utts | p5 | p25 | median | p75 | p95 | median within-utt sd |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reversed(gate["per_level"]):
        L.append(f"| {label(r['snr'])} | {r['mean']:.3f} | {r['sd_between_utts']:.3f} | {r['p5']:.3f} | "
                 f"{r['p25']:.3f} | {r['median']:.3f} | {r['p75']:.3f} | {r['p95']:.3f} | "
                 f"{r['median_within_utt_sd']:.3f} |")
    s = gate["per_utt_spearman"]
    L += ["",
          f"- Per-utterance Spearman rho (level vs weight, 8 points each): median {s['median']:.3f}, "
          f"positive in {100 * s['frac_positive']:.1f} %, rho = 1 in {100 * s['frac_rho_1']:.1f} %, min {s['min']:.3f}.",
          f"- Strictly monotone across all 8 levels: {100 * gate['frac_strictly_monotone']:.1f} % of utterances.",
          f"- Share of utterance-mean variance explained by the level (eta^2): {gate['eta2_level']:.3f}.",
          f"- Pooled Spearman over 3,200 (utterance, level) points: {gate['pooled_spearman']:.3f} (descriptive; points are not independent).",
          f"- AUC clean vs -10 dB: {gate['auc_clean_vs_minus10']:.3f}.", "",
          "| adjacent pair (lo -> hi) | paired: share with w(hi) > w(lo) | unpaired AUC |", "|---|---:|---:|"]
    for a in gate["adjacent"]:
        L.append(f"| {label(a['lo'])} -> {label(a['hi'])} | {100 * a['paired_frac_hi_gt_lo']:.1f} % | {a['auc_unpaired']:.3f} |")
    L += ["", "## Per-frame traces (gated arm; static arm verified 0.5 at every frame)", "",
          "| utterance | frames | " + " | ".join(label(s) for s in ASC) + " |",
          "|---|---:|" + "---|" * len(ASC)]
    for utt in trace_ids:
        cells = [f"{traces[utt][s]['mean']:.2f} +/- {traces[utt][s]['sd']:.2f} [{traces[utt][s]['min']:.2f}, {traces[utt][s]['max']:.2f}]"
                 for s in ASC]
        L.append(f"| {utt} | {traces[utt][ASC[0]]['n']} | " + " | ".join(cells) + " |")
    L += ["", "Cells: mean +/- sd [min, max] of the per-frame audio weight.", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="exp/per_utt_sweep")
    ap.add_argument("--static", default="rohan_avsr_3448_static")
    ap.add_argument("--gated", default="rohan_avsr_3448_gated")
    args = ap.parse_args()

    data = load(args.dir, args.static, args.gated)
    checks = integrity(data)
    table = cer_table(data)
    gate, W = gate_analysis(data)
    trace_ids, traces = trace_summary(data)

    plot_gate(W, os.path.join(args.dir, "gate_vs_snr_per_utt.png"))
    plot_delta(table, os.path.join(args.dir, "delta_cer_bootstrap.png"))
    plot_traces(data, trace_ids, os.path.join(args.dir, "gate_traces.png"))
    write_markdown(os.path.join(args.dir, "summary.md"), checks, table, gate, traces, trace_ids)
    with open(os.path.join(args.dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"checks": checks, "cer": table, "gate": gate,
                   "traces": {u: {str(k): v for k, v in t.items()} for u, t in traces.items()}}, f, indent=2)
    print(open(os.path.join(args.dir, "summary.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
