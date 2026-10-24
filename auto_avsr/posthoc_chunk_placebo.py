"""POST HOC, exploratory (written after the section 5.2 Part B results; NOT part of the
pre-registered analysis or its verdict). Why is the placebo difference not ~0?

Hypothesis: uniform chunk-start placement (start ~ U[0, n - len]) under-covers the
first and last ~len frames of each utterance, which are mostly leading/trailing
silence, where the clean gate weight is lower (Part A). 'Neither' frames are then
enriched with low-w edge frames, so w(chunk) - w(neither) > 0 without corruption.

Reads the gated arm's frames.npz only and prints; writes nothing.

    python posthoc_chunk_placebo.py --dir exp/chunk_sweep
"""
import argparse
import os

import numpy as np

EDGE = 25  # = the longest chunk; frames closer than this to an edge are under-covered
BINS = (("0-4", 0, 5), ("5-9", 5, 10), ("10-24", 10, 25), (">=25", 25, 10**9))


def load(path):
    z = np.load(path)
    o = z["offsets"]
    return [(z["gate_w_audio"][o[i]:o[i + 1]], z["audio_mask"][o[i]:o[i + 1]],
             z["video_mask"][o[i]:o[i + 1]]) for i in range(len(o) - 1)]


def by_edge_distance(rows):
    w, cov = {b: [] for b, _, _ in BINS}, {b: [] for b, _, _ in BINS}
    for g, a, v in rows:
        n = len(g)
        d = np.minimum(np.arange(n), n - 1 - np.arange(n))
        for b, lo, hi in BINS:
            m = (d >= lo) & (d < hi)
            w[b].append(g[m].mean())
            cov[b].append((a | v)[m].mean())
    return {b: (np.median(w[b]), np.mean(cov[b])) for b, _, _ in BINS}


def d_medians(rows, interior):
    """Four-category reading (as in both_chunks), k = 0: median per-utterance
    w(audio-only) - w(neither) and w(video-only) - w(neither)."""
    out = {"dA": [], "dV": []}
    for g, a, v in rows:
        keep = np.ones(len(g), bool)
        if interior:
            keep[:EDGE] = False
            keep[len(g) - EDGE:] = False
        nei = keep & ~a & ~v
        for name, m in (("dA", keep & a & ~v), ("dV", keep & v & ~a)):
            if m.any() and nei.any():
                out[name].append(g[m].mean() - g[nei].mean())
    return {k: (float(np.median(x)), len(x)) for k, x in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="exp/chunk_sweep")
    ap.add_argument("--exp", default="rohan_avsr_3448_gated")
    args = ap.parse_args()
    run = lambda c: load(os.path.join(args.dir, args.exp, c, "frames.npz"))
    pl = run("placebo")

    print("POST HOC (not pre-registered). Gated arm, placebo (= clean input).")
    print("distance from nearest utterance edge (frames) | median per-utt w | mean P(frame in any mask)")
    for b, (w, c) in by_edge_distance(pl).items():
        print(f"  {b:>6} | {w:.3f} | {c:.3f}")
    print(f"\nmedian d, k = 0, four-category reading [n utterances]; interior = frames >= {EDGE} from both edges")
    for cond, rows in (("placebo", pl), ("audio_chunks", run("audio_chunks")), ("video_chunks", run("video_chunks"))):
        for label, interior in (("all frames", False), ("interior", True)):
            r = d_medians(rows, interior)
            print(f"  {cond:>12} {label:>10}: " + "  ".join(f"{k} {m:+.4f} [{n}]" for k, (m, n) in r.items()))


if __name__ == "__main__":
    main()
