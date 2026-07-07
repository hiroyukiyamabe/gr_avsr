#!/usr/bin/env python
"""Plot an SNR sweep produced by eval.py's per-experiment CSVs.

Each eval run appends a row (snr_target, cer|wer, gate_w_audio) to
``snr_sweep_<exp_name>.csv``. Point this script at the baseline and gated CSVs
to render two panels:

  (left)  error rate  vs SNR  -- lower & flatter = more noise-robust
  (right) audio gate weight vs SNR -- should rise with SNR for the gated model
          and sit flat at 0.5 for the neutral-gate baseline

Usage:
  python plot_snr_sweep.py \
      --baseline exp/snr_sweep_rohan_avsr.csv \
      --gated    exp/snr_sweep_rohan_avsr_gated.csv \
      --out      exp/snr_sweep.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # headless: write a file, don't open a window
import matplotlib.pyplot as plt

# The eval config uses 999999 dB as the "clean" (no-noise) sentinel.
CLEAN_SENTINEL = 100000

# Colourblind-safe: blue (baseline) vs vermillion (gated).
C_BASELINE = "#0072B2"
C_GATED = "#D55E00"


def read_sweep(path):
    """Return (finite_rows, clean_row) sorted by SNR.

    Each row is a dict with float 'snr', 'err' (may be None), 'gate' (may be None).
    The clean sentinel row (if any) is returned separately so it can be drawn at
    the right edge with a 'clean' tick.
    """
    finite, clean = [], None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        err_key = "cer" if "cer" in reader.fieldnames else "wer"
        for r in reader:
            def num(key):
                v = r.get(key, "")
                return float(v) if v not in ("", None) else None
            row = {"snr": float(r["snr_target"]), "err": num(err_key), "gate": num("gate_w_audio")}
            if row["snr"] >= CLEAN_SENTINEL:
                clean = row
            else:
                finite.append(row)
    finite.sort(key=lambda d: d["snr"])
    return finite, err_key, clean


def series(finite, clean, field, clean_x):
    """x/y lists for a field ('err' or 'gate'), skipping missing values.

    The clean point (if present) is appended at ``clean_x`` so it plots to the
    right of the finite SNRs.
    """
    xs, ys = [], []
    for row in finite:
        if row[field] is not None:
            xs.append(row["snr"])
            ys.append(row[field])
    if clean is not None and clean[field] is not None:
        xs.append(clean_x)
        ys.append(clean[field])
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", help="snr_sweep_<baseline>.csv (static fusion)")
    ap.add_argument("--gated", help="snr_sweep_<gated>.csv (dynamic fusion)")
    ap.add_argument("--out", default="snr_sweep.png")
    args = ap.parse_args()

    runs = []  # (label, colour, finite, err_key, clean)
    if args.baseline:
        f, ek, c = read_sweep(args.baseline)
        runs.append(("static (baseline)", C_BASELINE, f, ek, c))
    if args.gated:
        f, ek, c = read_sweep(args.gated)
        runs.append(("gated (dynamic)", C_GATED, f, ek, c))
    if not runs:
        ap.error("pass at least one of --baseline / --gated")

    err_label = "CER" if any(r[3] == "cer" for r in runs) else "WER"

    # Place the "clean" point one grid step to the right of the largest finite SNR.
    all_finite_snrs = [row["snr"] for _, _, finite, _, _ in runs for row in finite]
    step = 5
    clean_x = (max(all_finite_snrs) + step) if all_finite_snrs else 25

    fig, (ax_err, ax_gate) = plt.subplots(1, 2, figsize=(11, 4.2))

    for label, colour, finite, _, clean in runs:
        xs, ys = series(finite, clean, "err", clean_x)
        if xs:
            ax_err.plot(xs, [y * 100 for y in ys], "o-", color=colour, label=label)
        gx, gy = series(finite, clean, "gate", clean_x)
        if gx:
            ax_gate.plot(gx, gy, "o-", color=colour, label=label)

    # Shared x formatting: label the clean sentinel position as "clean".
    for ax in (ax_err, ax_gate):
        ax.set_xlabel("SNR (dB)")
        ticks = sorted(set(all_finite_snrs))
        if any(r[4] is not None for r in runs):
            ax.set_xticks(ticks + [clean_x])
            ax.set_xticklabels([str(int(t)) for t in ticks] + ["clean"])
        else:
            ax.set_xticks(ticks)
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax_err.set_ylabel(f"{err_label} (%)")
    ax_err.set_title(f"{err_label} vs SNR (lower & flatter = more robust)")

    ax_gate.set_ylabel("mean audio gate weight")
    ax_gate.set_title("Audio reliance vs SNR")
    ax_gate.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
    ax_gate.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
