"""POST HOC, exploratory; not pre-registered, not part of any verdict (written
2026-10-24, after the section 5.2 Part B results; HANDOVER_CHANGES.md Task 3).
Is the gated arm worse *because of* the chunk corruption, or only by its clean
offset?

Estimand per condition c (CER pt), twin gap minus the placebo (= clean) gap:

    I(c) = [CER_gated(c) - CER_static(c)] - [CER_gated(placebo) - CER_static(placebo)]

CER = sum edit_distance / sum ref_len over the utterances in the sample, x 100.
Bootstrap: B = 10,000 utterance resamples from np.random.default_rng(0), ONE
resample shared by all arms and conditions; 95 % percentile CI. The CI covers
test-utterance sampling only, not training-seed variance.

Reads the twins' rows.csv only; prints the table and writes a new markdown file
(refuses to overwrite).

    python posthoc_cer_interaction.py --dir exp/chunk_sweep
"""
import argparse
import csv
import os

import numpy as np

ARMS = ("rohan_avsr_3448_static", "rohan_avsr_3448_gated")
CONDITIONS = ("audio_chunks", "video_chunks", "both_chunks")
B = 10_000
# Raw twin gap minus the clean gap, RESULTS.md section 2.4 CER table
# (+0.88 - 0.39, +0.41 - 0.39, +0.95 - 0.39 after rounding).
EXPECTED = {"audio_chunks": 0.49, "video_chunks": 0.02, "both_chunks": 0.56}
TOL = 0.01


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return ([r["utt_id"] for r in rows],
            np.array([int(r["edit_distance"]) for r in rows]),
            np.array([int(r["ref_len"]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="exp/chunk_sweep")
    ap.add_argument("--out", default="exp/chunk_sweep/posthoc_cer_interaction.md")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")

    edits, ref = {}, None
    utts = None
    for arm in ARMS:
        for cond in ("placebo",) + CONDITIONS:
            u, e, r = load(os.path.join(args.dir, arm, cond, "rows.csv"))
            if utts is None:
                utts, ref = u, r
            assert u == utts, f"utt_id order differs: {arm}/{cond}"
            assert np.array_equal(r, ref), f"ref_len differs: {arm}/{cond}"
            edits[arm, cond] = e
    n = len(utts)

    idx = np.random.default_rng(0).integers(0, n, size=(B, n))
    ref_b = ref[idx].sum(axis=1)

    def cer(arm, cond):
        e = edits[arm, cond]
        return 100 * e.sum() / ref.sum(), 100 * e[idx].sum(axis=1) / ref_b

    def gap(cond):
        (gs, gb), (ss, sb) = cer(ARMS[1], cond), cer(ARMS[0], cond)
        return gs - ss, gb - sb

    clean, clean_b = gap("placebo")
    lines = [
        "# POST HOC: CER interaction, gated vs static, net of the clean offset",
        "",
        "POST HOC, exploratory; not pre-registered, not part of any verdict.",
        f"I(c) = [gated - static](c) - [gated - static](placebo), CER pt; n = {n} utterances;",
        f"paired bootstrap B = {B}, np.random.default_rng(0), one resample shared by all",
        "arms and conditions; 95 % percentile CI. The CI covers test-utterance sampling",
        "only, not training-seed variance.",
        "",
        f"Clean (placebo) gap: {clean:+.2f} pt "
        f"[{np.percentile(clean_b, 2.5):+.2f}, {np.percentile(clean_b, 97.5):+.2f}]",
        "",
        "| condition | gated - static | minus clean gap = I(c) [95 % CI] |",
        "|---|---:|---|",
    ]
    for cond in CONDITIONS:
        g, g_b = gap(cond)
        i, i_b = g - clean, g_b - clean_b
        lo, hi = np.percentile(i_b, [2.5, 97.5])
        if abs(i - EXPECTED[cond]) > TOL:
            raise SystemExit(f"{cond}: I = {i:+.4f}, expected {EXPECTED[cond]:+.2f} +- {TOL}; abort")
        lines.append(f"| {cond} | {g:+.2f} | {i:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "Command (from auto_avsr/): `python posthoc_cer_interaction.py --dir exp/chunk_sweep`", ""]

    text = "\n".join(lines)
    print(text)
    with open(args.out, "x", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
