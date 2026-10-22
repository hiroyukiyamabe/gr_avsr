# Results: Tamura Reproduction + Dynamic-Fusion Study

Final numbers as of 2026-07-12 (extended SNR sweep added 2026-10-08, §2.1).
Reproduction target: Kondo & Tamura, "Audio-Visual Speech Recognition Based on
Cross-Lingual Transfer Learning" (APSIPA ASC 2025),
framework (c) — English front-ends from the 3,448 h Auto-AVSR checkpoints,
fine-tuned; everything else trained from scratch on Japanese ROHAN4600 data.
All CER in % on our 400-sentence test split, `model_avg_10.pth` checkpoints,
white-noise evals (NoiseX-92, held out from training). Full experimental log:
`REPRODUCTION_PLAN.md`.

## 1. Main reproduction table (framework (c))

| Row | Condition | Ours | Paper | Verdict / note |
|---|---|---:|---:|---|
| ASR (16 kHz) | clean | **3.87** | 3.88 | exact reproduction |
| ASR (16 kHz) | white 10 dB | **26.09** | 26.02 | near-exact |
| ASR (16 kHz) | white 5 dB | **43.42** | 40.13 | close; high-variance regime |
| VSR (25 fps) | clean | **26.36** | 19.70 | gap isolated to video data (§3) |
| AVSR static | clean | **3.13** | 2.78 ⁽¹⁾ | matches 16 kHz-adjusted expectation ~3.2–3.3 ⁽²⁾ |
| AVSR static | white 10 dB | **10.79** | 11.95 ⁽¹⁾ | at/better than paper ⁽³⁾ |
| AVSR static | white 5 dB | **15.73** | 16.11 ⁽¹⁾ | matched |

⁽¹⁾ Paper's AVSR is 19,200 Hz-only; ours runs 16,000 Hz (see deviations, §5).
⁽²⁾ Adjustment via the paper's own ASR 16 k vs 19.2 k penalty (3.88 vs 3.34).
⁽³⁾ Direction consistent with the paper's ASR rows: 16 kHz is the more
noise-robust rate (their ASR @ 10 dB: 26.02 @ 16 k vs 34.74 @ 19.2 k).

**Relative multimodal gains reproduce:** clean ASR→AVSR −19 % rel. (paper −17 %);
10 dB −59 % rel. (paper −54 %); 5 dB −64 % rel. (paper −60 %). The 10→5 dB
degradation asymmetry also reproduces: AVSR +4.9 pt vs ASR +17.3 pt (paper:
+4.2 vs +14.1).

## 2. Dynamic-fusion twin study (blind per-frame gate, Option A)

Twin runs identical in every respect (16 kHz, `max_frames` 800, 40 epochs,
3,448 h AV front-ends) except `fusion_gate_freeze`.

Noisy numbers in this table are the July 2026 evals (`eval.py`). The October
re-eval of the same checkpoints (§2.1) gives slightly different noisy CERs
because eval noise is unseeded — the 10 dB winner flips. Read both tables
together; neither set of sub-0.5 pt gaps is significant.

| Condition | AVSR static (frozen gate) | AVSR gated | gate_w_audio (gated) |
|---|---:|---:|---:|
| clean | **3.13** | 3.52 | 0.537 |
| white 10 dB | 10.79 | **10.70** | 0.489 |
| white 5 dB | **15.73** | 16.28 | 0.443 |

**Verdict: mechanism confirmed, benefit absent.** The gate re-weights exactly as
designed — its audio weight falls monotonically with SNR with no SNR
supervision (static arm reads 0.5 exactly, the frozen-gate sanity check) — but
recognition ties at 10 dB (Δ0.09) and static wins clean (+0.39) and 5 dB
(+0.55). Interpretation: noise-augmented training already makes static fusion
robust; stationary white noise gives per-frame gating nothing temporal to
exploit; and even 5 dB audio out-informs clean video (AVSR 15.7 vs VSR 26.4),
so down-weighting audio has a price. **Future work:** non-stationary noise
(held-out babble), utterance-level gating. (SNR ≤ 0 dB: done, §2.1.)

### 2.1 Extended white-noise sweep (2026-10-08): clean → −10 dB

Same two `model_avg_10.pth` checkpoints, same eval settings as above
(`rohan_avsr_fixed`, 16 kHz, NoiseX-92 white via `eval_sweep.py`), extended to
eight levels. CSVs: `auto_avsr/exp/snr_sweep_rohan_avsr_3448_{static,gated}.csv`;
plot: `auto_avsr/exp/snr_sweep_white_8lvl.png`; log:
`auto_avsr/exp/white_sweep_8lvl.log` (all 16 evals, each delimited by
`=== <timestamp> <exp> snr=<snr> ===` / `=== exit <code> ===`; the
`NativeCommandError` "triton not found" lines are benign stderr).
CSV columns: `snr_target` (999999 = clean), `cer` (fraction, ×100 for %),
`gate_w_audio` (audio gate weight averaged over all test-set frames, `lightning_av.py`;
0.5 = frozen gate).

To reproduce (PowerShell, from `auto_avsr\`, venv active; ~12 min/eval,
~3.2 h total). `eval_sweep.py` **appends** to `exp/snr_sweep_<exp_name>.csv`,
so rename the existing CSVs first:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$log = "exp\white_sweep_8lvl.log"
foreach ($exp in "rohan_avsr_3448_static", "rohan_avsr_3448_gated") {
  foreach ($snr in 999999, 20, 15, 10, 5, 0, -5, -10) {
    "=== $(Get-Date -Format s) $exp snr=$snr ===" | Out-File -Append -Encoding utf8 $log
    python eval_sweep.py +experiment=rohan_avsr_fixed exp_dir=exp exp_name=$exp `
      pretrained_model_path=exp/$exp/model_avg_10.pth transfer_frontend=false `
      +data.num_workers=2 decode.snr_target=$snr `
      decode.noise_filename=../noise_datasets/white_16k.wav *>> $log
    "=== exit $LASTEXITCODE ===" | Out-File -Append -Encoding utf8 $log
  }
}
python plot_snr_sweep.py --baseline exp/snr_sweep_rohan_avsr_3448_static.csv `
  --gated exp/snr_sweep_rohan_avsr_3448_gated.csv --out exp/snr_sweep_white_8lvl.png
```

Expect noisy CERs to differ by a few tenths from the table (unseeded noise,
finding 2); clean must reproduce exactly (3.13 / 3.52).

| SNR (white) | AVSR static | AVSR gated | Δ (gated − static) | gate_w_audio (gated) |
|---|---:|---:|---:|---:|
| clean | **3.13** | 3.52 | +0.39 | 0.537 |
| 20 dB | 5.60 | **5.36** | −0.24 | 0.538 |
| 15 dB | 7.91 | **7.37** | −0.54 | 0.522 |
| 10 dB | **10.48** | 10.73 | +0.25 | 0.489 |
| 5 dB | **15.77** | 16.14 | +0.37 | 0.443 |
| 0 dB | **23.21** | 24.10 | +0.89 | 0.392 |
| −5 dB | 30.08 | **29.94** | −0.14 | 0.359 |
| −10 dB | 32.57 | **32.04** | −0.53 | 0.346 |

Reference: VSR (video only) 26.36 %.

**Findings.**

1. **Still no gated benefit at low SNR.** The winner alternates 4–4 across
   the eight levels; the mean Δ is +0.06 pt. The largest gap (static +0.89 at
   0 dB) is a single level, unreplicated.
2. **Eval noise is not seeded.** `AddNoise` draws a random offset into the
   noise file per utterance, so re-evaluating the same checkpoint changes
   noisy CERs: static 10 dB 10.79 → 10.48, 5 dB 15.73 → 15.77. Clean is
   bit-for-bit repeatable (3.13 / 3.52, gate 0.537), confirming the noise
   draw is the only eval-side randomness. At 10 dB the twin verdict *flipped*
   between July (gated ahead by 0.09) and October (static ahead by 0.25):
   gaps of ≲ 0.5 pt are not interpretable. Training-run variance (one seed
   per arm) is unmeasured and also affects clean.
3. **Fusion is worse than video alone below ~0 dB.** Both arms cross the VSR
   level (26.36 %) between 0 and −5 dB and plateau near 32 % at −10 dB: a
   ~6 pt "fallback gap" that neither fusion recovers.
4. **The gate detects but cannot act.** Its audio weight keeps falling with
   SNR but saturates near 0.35 (audio still enters at 2 × 0.35 = 0.7× scale).
   Likely causes, from the architecture: the gate sees the same inputs as the
   fusion MLP (no new information); scaling shrinks noisy features but does
   not remove them; the fusion MLP is the last stage before CTC and the
   decoder, so there is no video-only path around it; and training (babble,
   −5…20 dB, audio always present) never rewarded relying on video alone.
   The last point is a hypothesis; a forced-weight eval (pin w_a = 0 at
   −10 dB) would test it.

**Implications for follow-up work:** seed the eval noise per utterance ID so
both arms see identical noise, then report paired bootstrap intervals; a
reliability-gated *visual-only* path (cf. router-gated cross-attention,
arXiv 2508.18734) or modality dropout targets the fallback gap directly.

## 3. VSR ablations (why the 26.4 vs 19.7 gap is the video data)

| Variant | Frontend | Framing (mouth width in 96 px) | CER |
|---|---|---|---:|
| canonical | AV ckpt (3,448 h) | mouth crop 200→96 (~53 px) | **26.36** |
| frontend ablation | VSR ckpt (3,448 h) | mouth crop 200→96 (~53 px) | 26.63 |
| geometry ablation | AV ckpt | canonical recrop 234→96 (~45 px) | 26.50 |
| paper's own prep | VSR ckpt | whole-frame 300→96 (~31 px) | 28.04 |
| (historical) | VSR ckpt (438 h) | mouth crop | 27.26 |

VSR is ~26.4 % regardless of frontend scale, frontend modality, or crop
geometry — including the paper's own preprocessing, which is *worse* on our
recording. Every audio-bearing row in §1 reproduces, so the pipeline is
faithful; the gap is attributed to the recording: the papers used a public
ROHAN multimodal release (4,200 videos public as of Feb 2024; speaker
unresolved) whose duration fingerprint (8.15 s/utt) differs from our zundamon
recording (6.70 s/utt) — ~22 % slower speech = more frames per mora. The
corpus's own landmark audit confirms our framing is sound (mouth always inside
the crop; batch 1201–1600's landmark CSVs are broken, videos fine).

## 4. Run provenance (all validated in logged hparams)

Train log = `auto_avsr/exp/lightning_logs/version_N/`; eval log =
`auto_avsr/lightning_logs/version_N/`. Version numbers overlap between the two
dirs (e.g. train version_31 ≠ eval version_31). Each `hparams.yaml` records the
checkpoint, dataset root, `snr_target` and `noise_filename` actually used.

| Result | Train log | Eval log |
|---|---|---|
| ASR clean / noise | (2026-07-08 run) | —, version_20, version_21 |
| VSR canonical / vsr-fe / recrop / paper-prep | version_31 / 32 / 33 / 36 | version_10 / 11 / 12 / 13 |
| AVSR static clean / 10 dB / 5 dB | version_41 | version_14 / 18 / 19 |
| AVSR gated clean / 10 dB / 5 dB | version_42 | version_15 / 16 / 17 |
| AVSR static 8-level sweep (clean, 20 … −10 dB) | version_41 | version_22 – 29 |
| AVSR gated 8-level sweep (clean, 20 … −10 dB) | version_42 | version_30 – 37 |

## 5. Documented deviations from the paper

1. **Recording**: our zundamon ROHAN4600 videos (4,600, complete release) vs
   their unresolved public release (4,200 as of Feb 2024); split 3,800/400/400
   vs 3,400/400/400. The sole unresolved confound; affects VSR most.
2. **AVSR audio 16,000 Hz** (rate_ratio 640) vs their 19,200 Hz (768): the
   19.2 kHz path is implemented and unit-verified (`aux_rate_ratio`,
   parameter-free avgpool 20→24, `rohan_preprocessed_avsr19k2`), but at
   `max_frames` 1000 fp32 the AVSR model spills ~5.5 GB past the 24 GB RTX 4090
   under Windows/WDDM *at either rate*; arms ran at 16 kHz for stability.
3. **Batch spec**: VSR `max_frames` 1000 vs paper 1600 (doesn't fit, same
   memory ceiling); AVSR 800 vs paper 1000 (mid-run WDDM paging degradation at
   1000). Twin arms share every setting, so the fusion comparison is unaffected.
4. **Video framing**: mouth-centred 200→96 crop vs their whole-frame 300→96
   resize — tested head-to-head (§3); ours is 1.7 pt better on our data.
5. Audio for ASR/VSR label CSVs etc. identical to the paper's recipe: AdamW,
   peak LR 1e-4, 5-epoch warmup, cosine, wd 0.03, hybrid CTC/attention α=0.1,
   87 katakana IDs, grayscale + Normalize(0.421, 0.165), single GPU.
