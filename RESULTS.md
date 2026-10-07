# Results: Tamura Reproduction + Dynamic-Fusion Study

Final numbers as of 2026-07-12. Reproduction target: Kondo & Tamura, "Audio-Visual
Speech Recognition Based on Cross-Lingual Transfer Learning" (APSIPA ASC 2025),
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
(held-out babble), SNR ≤ 0 dB, utterance-level gating.

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

| Result | Train log | Eval log |
|---|---|---|
| ASR clean / noise | (2026-07-08 run) | —, version_20, version_21 |
| VSR canonical / vsr-fe / recrop / paper-prep | version_31 / 32 / 33 / 36 | version_10 / 11 / 12 / 13 |
| AVSR static clean / 10 dB / 5 dB | version_41 | version_14 / 18 / 19 |
| AVSR gated clean / 10 dB / 5 dB | version_42 | version_15 / 16 / 17 |

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
