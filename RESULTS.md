# Results: Tamura Reproduction + Dynamic-Fusion Study

Final numbers as of 2026-07-12 (extended SNR sweep added 2026-10-08, §2.1;
ASR/VSR under the same sweep 2026-10-22, §2.2; seeded per-utterance twin sweep
with paired bootstrap and gate diagnostic 2026-10-23, §2.3).
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
**Superseded for twin comparisons (2026-10-23) by §2.3**, where both arms see
identical seeded noise and every gap carries a paired-bootstrap interval. The
numbers below are kept as the July record.

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
both arms see identical noise, then report paired bootstrap intervals (**done
2026-10-23, §2.3**); a
reliability-gated *visual-only* path (cf. router-gated cross-attention,
arXiv 2508.18734) or modality dropout targets the fallback gap directly.

### 2.2 Single-modality references under the same sweep (2026-10-22)

ASR and VSR `model_avg_10.pth` checkpoints (`rohan_asr_3448`,
`rohan_vsr_3448`) evaluated under the same eight white-noise levels and eval
settings as §2.1 (`eval_sweep.py`, `white_16k.wav`, fixed dataset). CSVs:
`auto_avsr/exp/snr_sweep_rohan_{asr,vsr}_3448.csv`; combined plot (all four
branches): `auto_avsr/exp/snr_sweep_white_8lvl_all.png`; log:
`auto_avsr/exp/unimodal_sweep_8lvl.log`. AVSR columns are from §2.1.

| SNR (white) | ASR | VSR | AVSR static | AVSR gated | static − best single | ASR → AVSR static (rel.) |
|---|---:|---:|---:|---:|---:|---:|
| clean | 3.87 | 26.36 | **3.13** | 3.52 | −0.74 | −19 % |
| 20 dB | 9.78 | 26.36 | 5.60 | **5.36** | −4.18 | −43 % |
| 15 dB | 15.99 | 26.36 | 7.91 | **7.37** | −8.08 | −51 % |
| 10 dB | 26.01 | 26.36 | **10.48** | 10.73 | −15.53 | −60 % |
| 5 dB | 43.15 | 26.36 | **15.77** | 16.14 | −10.59 | −63 % |
| 0 dB | 73.89 | 26.36 | **23.21** | 24.10 | −3.15 | −69 % |
| −5 dB | 91.36 | **26.36** | 30.08 | 29.94 | +3.72 | −67 % |
| −10 dB | 95.60 | **26.36** | 32.57 | 32.04 | +6.21 | −66 % |

"Best single" = min(ASR, VSR) at that level; negative = fusion beats both
single-modality systems.

**Findings.**

1. **VSR is noise-invariant, now measured.** It scores 26.36 % at all eight
   levels, bit-identical. This is expected: the eval noise is added to audio
   only, and the video-only dataset path never loads audio
   (`datamodule/av_dataset.py`). The VSR line is a fixed reference.
2. **ASR collapses, crossing VSR at ~10 dB.** 3.87 → 95.60 %. At 10 dB, ASR
   (26.01) and VSR (26.36) tie; below that, audio alone is worse than video
   alone.
3. **Fusion beats both modalities from clean to 0 dB.** The largest gain over
   the better single modality is at 10 dB (−15.5 pt), exactly where the two
   single-modality systems tie, the classic complementarity point. AVSR
   crosses the VSR line between 0 and −5 dB, about 10 dB lower than ASR does.
4. **Below 0 dB, fusion is worse than video alone (fallback gap).** At
   −10 dB the audio stream is nearly uninformative (ASR 95.6 %), yet AVSR
   static is 6.2 pt worse than VSR. The gate recovers only 0.5 pt of that
   (gated 32.04). A reliability-aware fusion should do no worse than the
   better single modality, and both arms fail this below 0 dB. This confirms
   §2.1 finding 3 against the paired VSR eval. The relative ASR→AVSR gain
   flattens at −5/−10 dB only because ASR is saturating near 100 %.
5. **ASR rows of §1 hold.** Clean is bit-identical (3.87). The noisy re-evals
   (10 dB 26.09 → 26.01, 5 dB 43.42 → 43.15) are within the unseeded-noise
   spread noted in §2.1.

### 2.3 Per-utterance twin sweep, seeded noise (2026-10-23)

Phase 2c, **weak form** (CLAUDE.md §5.1): utterance-level gate weight against
target SNR, plus paired bootstrap of the twin CER gap. Inference only, same two
frozen `model_avg_10.pth` checkpoints as §2.1 (static sha256 `f543646f…`, gated
`6f0291cb…`), `rohan_avsr_fixed`, 16 kHz, NoiseX-92 white (`white_16k.wav`,
held out from training), eight levels, `decode.noise_seed=0`. Static arm
evaluated with `fusion_gate_freeze=true` (its training setting), gated with
`false`; nothing else differs.

**What changed vs §2.1.** (1) Eval noise is seeded: the crop offset into the
noise file is a fixed function of (`noise_seed`, utterance ID), so both arms
see bit-identical noisy audio (verified by a SHA-1 of the model's audio input,
400/400 utterances at every level), and each utterance uses the **same crop at
every SNR** (only the scale changes). Training noise is untouched (still
random). (2) `eval_sweep_utt.py` logs one row per (arm, SNR, utterance) plus
per-frame traces for 8 utterances fixed before the run (length quantiles; list
in `configs/decode/default.yaml`). It wraps the unchanged `test_step`, so
aggregates come from the same code: clean CER and gate are **bit-identical** to
§2.1 (static 0.031336478888988495, gated 0.03524336591362953 / 0.5370424389839172).

Outputs: `auto_avsr/exp/per_utt_sweep/` — `<exp_name>/snr_<snr>.csv` (columns
`utt_id, n_frames, gate_w_audio_mean/std/min/max, edit_distance, ref_len,
audio_sha1`), `*_traces.json`, `*_meta.json` (checkpoint + noise hashes,
logged and row-summed aggregates, eval log dir), `summary.md`/`summary.json`
(all tables below + 132 integrity checks), plots `gate_vs_snr_per_utt.png`,
`gate_traces.png`, `delta_cer_bootstrap.png`; log `per_utt_sweep.log`.

To reproduce (PowerShell, from `auto_avsr\`, venv active; ~15 min/eval, ~4 h).
The script refuses to overwrite, so move `exp\per_utt_sweep` first:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$log = "exp\per_utt_sweep\per_utt_sweep.log"
$arms = @(@("rohan_avsr_3448_static", "true"), @("rohan_avsr_3448_gated", "false"))
foreach ($snr in 999999, 20, 15, 10, 5, 0, -5, -10) {
  foreach ($arm in $arms) {
    $exp = $arm[0]; $frz = $arm[1]
    "=== $(Get-Date -Format s) $exp snr=$snr ===" | Out-File -Append -Encoding utf8 $log
    python eval_sweep_utt.py +experiment=rohan_avsr_fixed exp_dir=exp exp_name=$exp `
      pretrained_model_path=exp/$exp/model_avg_10.pth transfer_frontend=false `
      model.audiovisual_backbone.fusion_gate_freeze=$frz `
      +data.num_workers=2 decode.snr_target=$snr decode.noise_seed=0 `
      decode.noise_filename=../noise_datasets/white_16k.wav *>> $log
    "=== exit $LASTEXITCODE ===" | Out-File -Append -Encoding utf8 $log
  }
}
python analyze_per_utt_sweep.py --dir exp/per_utt_sweep
```

**Integrity (all passed, `summary.md`).** 400 utterances per run; static arm
exactly 0.5 at every frame of every utterance (rows and traces); identical
audio across arms; noise applied at every finite level; rows sum exactly to the
logged totals; eval `hparams.yaml` of the two arms differ only in `exp_name`,
`pretrained_model_path`, `fusion_gate_freeze`, and across levels only in
`snr_target`.

**CER, seeded noise.** Δ = gated − static (pt); 95 % CI from a paired
bootstrap over utterances (B = 10,000, same resample for both arms).

| SNR (white) | AVSR static | AVSR gated | Δ | 95 % CI | utts gated better / worse / tie | gate_w_audio (gated) |
|---|---:|---:|---:|---|---|---:|
| clean | **3.13** | 3.52 | +0.39 | [+0.09, +0.69] | 61 / 89 / 250 | 0.537 |
| 20 dB | 5.46 | 5.36 | −0.10 | [−0.52, +0.34] | 117 / 95 / 188 | 0.538 |
| 15 dB | 8.00 | **7.24** | −0.77 | [−1.25, −0.30] | 151 / 107 / 142 | 0.522 |
| 10 dB | 10.64 | 10.40 | −0.24 | [−0.76, +0.29] | 146 / 137 / 117 | 0.489 |
| 5 dB | **15.40** | 16.30 | +0.90 | [+0.25, +1.55] | 140 / 176 / 84 | 0.443 |
| 0 dB | **23.19** | 24.11 | +0.92 | [+0.08, +1.73] | 153 / 191 / 56 | 0.392 |
| −5 dB | 30.11 | 30.13 | +0.02 | [−0.73, +0.78] | 153 / 147 / 100 | 0.359 |
| −10 dB | 32.85 | 32.21 | −0.64 | [−1.44, +0.15] | 172 / 147 / 81 | 0.346 |

Bold = the CI excludes 0. Eval logs: static `lightning_logs/version_54, 56, …, 68`,
gated `version_55, 57, …, 69` (clean → −10 dB).

**Gate vs target SNR (gated arm; utterance-mean audio weight over 400 utterances).**

| SNR | mean | sd between utts | p5–p95 | median within-utt sd |
|---|---:|---:|---|---:|
| clean | 0.539 | 0.020 | 0.509–0.571 | 0.123 |
| 20 dB | 0.539 | 0.020 | 0.507–0.571 | 0.094 |
| 15 dB | 0.523 | 0.020 | 0.491–0.555 | 0.091 |
| 10 dB | 0.490 | 0.018 | 0.462–0.519 | 0.089 |
| 5 dB | 0.443 | 0.016 | 0.417–0.472 | 0.077 |
| 0 dB | 0.392 | 0.014 | 0.369–0.415 | 0.058 |
| −5 dB | 0.358 | 0.012 | 0.340–0.378 | 0.052 |
| −10 dB | 0.345 | 0.011 | 0.328–0.364 | 0.054 |

**Findings.**

1. **Twin CER: still no consistent gated benefit.** With identical noise and
   paired intervals, four of eight gaps exclude zero, in both directions:
   static is better clean (+0.39), at 5 dB (+0.90) and 0 dB (+0.92); gated is
   better at 15 dB (−0.77). The other four include zero. These signs agree
   with every earlier unseeded eval of the same level (5 dB: July +0.55 and
   October +0.37; 0 dB: October +0.89; 15 dB: October −0.54), so they are
   probably not artefacts of this one noise draw. The 10 dB sign has now
   flipped three times (July −0.09, October +0.25, now −0.24; CI spans 0). The intervals cover test-utterance sampling for one
   noise draw and one training seed per arm only. Neither noise-draw variance
   (the seeded aggregates moved up to 0.4 pt vs §2.1, e.g. static 5 dB 15.77 →
   15.40) nor training-seed variance is included, so the true uncertainty is
   wider. With eight levels, about one CI would exclude zero by chance at 95 %.
2. **Weak form: the utterance-level gate weight is ordered by target SNR.**
   For **every** one of the 400 utterances, the mean audio weight is strictly
   higher at the higher SNR of each adjacent pair from −10 to 20 dB (100 % of
   utterances, all six pairs). This is a paired comparison under the same-crop
   design, so only the noise scale differs within an utterance. The level
   explains 95.4 % of the variance of utterance means (η²); per-utterance
   Spearman ρ across the 8 levels has median 0.976, min 0.929. Separation
   *across* utterances (unpaired AUC) is high in the middle (−5↔0: 0.97,
   0↔5: 0.99, 5↔10: 0.97) and weak at the ends (−10↔−5: 0.79, 15↔20: 0.72).
   20 dB and clean are indistinguishable (paired 46 %, AUC 0.50): the response
   saturates above ~20 dB and flattens below −5 dB. Pooled gate means match
   §2.1 to three decimals; white noise is stationary, so the noise draw barely
   moves them.
3. **The gate varies within utterances, more than between them.** The median
   per-utterance sd of the per-frame weight (0.05–0.12) is 4–6× the
   between-utterance sd of the means (0.01–0.02) at every level. It is
   largest on clean speech (0.12; trace ranges ≈ 0.32–0.80) and shrinks as SNR falls (0.05 at
   −10 dB, where the weight stays low throughout). In the traces
   (`gate_traces.png`), some stretches of each utterance converge to
   ≈ 0.40–0.45 at every SNR, while other stretches fan out by SNR. **Not
   verified:** these may be pauses or non-speech frames versus speech, but no
   speech-activity or segmental-SNR reference was computed.
4. **Scope.** This is the weak form only: one stationary noise type at a
   fixed target SNR per utterance. It shows that the utterance-mean weight
   tracks the utterance's noise level; it does **not** validate a per-frame
   reliability interpretation. Finding 3 shows the per-frame weight moves for
   reasons the target SNR does not explain. The strong form (per-frame weight
   against segmental SNR, or against a known per-frame corruption mask)
   remains open.

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
| ASR 8-level sweep (clean, 20 … −10 dB) | (2026-07-08 run) | version_38 – 45 |
| VSR 8-level sweep (clean, 20 … −10 dB) | version_31 | version_46 – 53 |
| AVSR per-utterance seeded sweep, static (clean … −10 dB) | version_41 | version_54, 56, 58, 60, 62, 64, 66, 68 |
| AVSR per-utterance seeded sweep, gated (clean … −10 dB) | version_42 | version_55, 57, 59, 61, 63, 65, 67, 69 |

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
