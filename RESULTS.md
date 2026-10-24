# Results: Tamura Reproduction + Dynamic-Fusion Study

Final numbers as of 2026-07-12 (extended SNR sweep added 2026-10-08, §2.1;
ASR/VSR under the same sweep 2026-10-22, §2.2; seeded per-utterance twin sweep
with paired bootstrap and gate diagnostic 2026-10-23, §2.3; strong-form gate
test with chunk-wise corruption 2026-10-23, §2.4).
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
(held-out babble), utterance-level gating. (SNR ≤ 0 dB: done, §2.1.
Chunk-wise, eval-only white noise and video corruption: done 2026-10-23,
§2.4. Static is still better, by 0.9 pt under audio chunks.)

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
   speech-activity or segmental-SNR reference was computed. (**Verified
   2026-10-23, §2.4 Part A:** they are pauses. The across-level sd is
   0.03–0.05 in non-speech vs 0.12–0.14 in speech frames.)
4. **Scope.** This is the weak form only: one stationary noise type at a
   fixed target SNR per utterance. It shows that the utterance-mean weight
   tracks the utterance's noise level; it does **not** validate a per-frame
   reliability interpretation. Finding 3 shows the per-frame weight moves for
   reasons the target SNR does not explain. The strong form (per-frame weight
   against segmental SNR, or against a known per-frame corruption mask)
   remains open. (**Done 2026-10-23, §2.4: FAIL under the pre-registered
   rule.** Finding 3's converging stretches are pauses, §2.4 Part A.)

### 2.4 Strong-form gate test: speech activity and chunk-wise corruption (2026-10-23)

Phase 2c, **strong form** (CLAUDE.md §5.2). Inference only, with the same two
frozen `model_avg_10.pth` checkpoints as §2.3 (static sha256 `f543646f…`,
gated `6f0291cb…`). `rohan_asr_3448` and `rohan_vsr_3448` `model_avg_10.pth`
serve as single-modality references. Every corruption setting, mask rule,
threshold and analysis choice was committed before the first real run
(`dba2502`). All Part B runs ran at `4f0c3eb` with a clean tree. The full
pre-registration is in the REPRODUCTION_PLAN.md strong-form entry.

**Verdict (pre-registered rule): FAIL.** The per-frame gate weight is **not**
validated as a reliability indicator. As CLAUDE.md §5.2 prescribes, the gate
is reported as reliability-sensitive at the utterance level only (§2.3).

The rule failed in two places. (1) The placebo control is not ≈ 0 for either
measure: random masks with no corruption already give an inside-vs-outside
difference. (2) For video, only 65 % of utterances have the expected sign
(the rule needs ≥ 75 %). The audio measure meets the CI and sign criteria and
fails on the placebo only.

A post-hoc check (below, not part of the verdict) traces the placebo offset to
chunk placement versus utterance-edge silence. The placebo-corrected and
chunk-locked readouts (descriptive, pre-registered outputs) show a local
response of the per-frame weight to corruption. They were not the decision
rule and do not overturn it.

#### Part A: speech activity and segmental SNR on the §2.3 traces (no GPU)

The 8 §2.3 trace utterances, gated arm, all 8 levels (`analyze_gate_partA.py`).
Each model input was rebuilt from (`noise_seed` 0, utterance ID) through
`AudioTransform`/`AddNoise` and matches the logged `audio_sha1` in 128/128
cases (8 utterances × 8 levels × 2 arms). Frame length equals video frames
equals label length for every utterance. The static traces are exactly 0.5.

- **VAD:** on clean audio, a frame is speech iff its energy ≥ utterance max − 30 dB. The threshold was fixed before looking; −20 and −40 dB are sensitivity checks only.
- **Segmental SNR:** computed per 640-sample frame from the noise actually added.
- **Outputs:** `auto_avsr/exp/per_utt_sweep/partA_summary.md`/`.json`, `partA_frames.csv`, `partA_traces.png`.

| SNR (white) | median w, non-speech | median w, speech | speech − non-speech (median; > 0 in) | Spearman ρ(w, segmental SNR), median [min, max] |
|---|---:|---:|---|---|
| clean | 0.443 | 0.639 | +0.194 (8/8) | +0.709 [+0.605, +0.734] ⁽¹⁾ |
| 20 dB | 0.471 | 0.610 | +0.131 (8/8) | +0.657 [+0.575, +0.721] |
| 15 dB | 0.464 | 0.589 | +0.120 (8/8) | +0.604 [+0.562, +0.702] |
| 10 dB | 0.445 | 0.549 | +0.096 (8/8) | +0.545 [+0.462, +0.659] |
| 5 dB | 0.423 | 0.474 | +0.043 (8/8) | +0.292 [+0.153, +0.546] |
| 0 dB | 0.404 | 0.389 | −0.017 (2/8) | −0.241 [−0.556, +0.038] |
| −5 dB | 0.393 | 0.333 | −0.061 (0/8) | −0.627 [−0.695, −0.517] |
| −10 dB | 0.389 | 0.307 | −0.081 (0/8) | −0.733 [−0.795, −0.613] |

⁽¹⁾ Clean has no noise, so this is ρ against frame speech energy.

**Findings (Part A).**

1. **The converging stretches of §2.3 finding 3 are pauses.** Take each
   frame's weight and compute its sd across the 8 levels. That sd is
   0.115–0.136 in speech frames and 0.026–0.045 in non-speech frames, per
   utterance. The per-utterance AUC (sd higher in speech) is 0.93–0.98.
   Non-speech weight stays within 0.38–0.49 at every level.
2. **In speech, the weight follows the utterance's noise level, in both
   directions.** Speech-frame weight runs from 0.31 (−10 dB) to 0.64 (clean).
   The speech − non-speech difference changes sign between 0 and 5 dB. This
   holds at all three VAD thresholds (−20/−30/−40 dB: −0.077/−0.081/−0.080 at
   −10 dB, +0.174/+0.194/+0.173 clean).
3. **The per-frame weight does not track segmental SNR monotonically.** At
   ≤ 0 dB the correlation is negative: within an utterance, speech frames have
   the higher segmental SNR but get the *lower* audio weight. Descriptively,
   the weight behaves like speech activity × utterance noise level. In pauses
   it sits near 0.4; in speech it moves up or down with the noise level.
4. **Scope (as pre-registered).** Under stationary white noise, segmental SNR
   is driven by speech energy, so it is confounded with speech activity.
   Part A cannot separate reliability from speech activity.

#### Part B: chunk-wise corruption with known per-frame masks

`decode.chunk_condition` adds seeded, eval-only corruption in the test
transforms. Draws are a function of (`noise_seed` 0, utterance ID, stream)
only. With the flag off, the pipelines are bit-identical to HEAD, including
RNG consumption.

- **Chunks:** per stream, 1–3 chunks of 10–25 frames (0.4–1.0 s) with a uniform start.
- **Audio:** NoiseX-92 white (`white_16k.wav`) at −5 dB relative to the whole clean utterance, added inside the chunks only, before `_instance_layer_norm`.
- **Video:** on the 88×88 mouth crop, each chunk is either Gaussian blur (σ = 4, kernel 25) or an occluder (p = ½ each).
  - The occluder is an object from the eval half of the NatOcc COCO set, 35×35 px.
  - It sits at the centre (42, 36) ± (20, 10) px, fixed for the whole chunk.
- **Eval-only from now on:** white noise, blur and the eval-half occluders.
- **Analysis:** boundary exclusion k = 2, the primary reading.
- **Coverage:** audio and video each corrupt 20 % of frames on average (0.207 / 0.200).
  - Frame categories: audio-only 15.8 %, video-only 15.0 %, both 4.0 %, neither 65.2 %.
  - The two streams overlap in 223/400 utterances.
  - Video-corrupted frames: 6,907 occluded, 6,391 blurred.

Runs:

- AVSR static and gated × {placebo, audio_chunks, video_chunks, both_chunks}.
- ASR × {placebo, audio_chunks}.
- VSR × {placebo, video_chunks}.
- Placebo draws and logs the both_chunks masks but corrupts nothing.

Outputs are in `auto_avsr/exp/chunk_sweep/`:

- Per run, `<exp_name>/<condition>/`: rows CSV, `frames.npz` (per-frame `gate_w_audio` and both masks for all 400 utterances) and `meta.json`.
- `summary.md`/`summary.json`: all tables below and 198 integrity checks.
- Plots: `gate_chunk_measures.png`, `gate_chunk_locked.png`, `gate_chunk_traces.png`.
- Log: `chunk_sweep.log`.

To reproduce, use the commands in the REPRODUCTION_PLAN.md strong-form
entry (~3 h).

**Integrity (all 198 passed).**

- All 12 runs used the same masks.
- The static arm reads exactly 0.5 at every frame in all four conditions.
- The twins saw bit-identical audio and video inputs (SHA-1, 400/400) in every condition.
- Corruption is applied only where the condition says.
- VSR video equals AVSR video.
- Placebo is bit-identical to the §2.3 clean run (inputs, edits, CER 0.031336478888988495 / 0.03524336591362953, gate 0.5 / 0.5370424389839172).
- ASR/VSR placebo equals §2.2 clean (3.87 / 26.36).
- Eval `hparams.yaml` of the twins differ only in `exp_name`, `pretrained_model_path` and `fusion_gate_freeze`; across conditions they differ only in `chunk_condition`.
- Every run used the same git HEAD with `dirty == false`.

**Gate response (gated arm).**

- **Measures:** dA = mean w in audio-only-corrupted frames − mean w in
  uncorrupted ("neither") frames, per utterance; expected < 0. dV = the same
  for video-only frames; expected > 0.
- **Statistics:** median [95 % bootstrap CI, B = 10,000]. "Share" is the
  share of utterances with the expected sign. AUC is the pooled frame AUC in
  the expected direction (0.5 = none).
- **Placebo:** the placebo run read with the same masks.
- **Corrected:** the per-utterance difference-in-differences against the
  placebo (same utterance, same frames), so effects present in both runs
  cancel.

| condition | measure | k | run: median [CI] | share | AUC | placebo: median [CI] | share | corrected: median [CI] | share |
|---|---|---:|---|---:|---:|---|---:|---|---:|
| audio_chunks | dA | 2 | **−0.078** [−0.082, −0.072] | 95 % | 0.675 | +0.037 [+0.024, +0.051] | 36 % | −0.115 [−0.131, −0.104] | 82 % |
| audio_chunks | dA | 0 | −0.072 [−0.075, −0.066] | 96 % | 0.662 | +0.030 [+0.023, +0.042] | 36 % | −0.104 [−0.114, −0.091] | 83 % |
| video_chunks | dV | 2 | **+0.058** [+0.042, +0.073] | 65 % | 0.604 | +0.029 [+0.013, +0.039] | 57 % | +0.027 [+0.024, +0.030] | 93 % |
| video_chunks | dV | 0 | +0.051 [+0.033, +0.064] | 65 % | 0.596 | +0.022 [+0.009, +0.036] | 57 % | +0.024 [+0.022, +0.027] | 93 % |
| both_chunks | dA | 2 | −0.069 [−0.075, −0.064] | 89 % | 0.658 | +0.051 [+0.036, +0.061] | 36 % | −0.118 [−0.134, −0.103] | 79 % |
| both_chunks | dV | 2 | +0.053 [+0.031, +0.075] | 65 % | 0.597 | +0.028 [+0.003, +0.045] | 56 % | +0.027 [+0.023, +0.030] | 86 % |

Bold = the two decision measures. **Rule (k = 2).** Each decision measure
must meet three criteria:

- Its CI excludes 0 on the expected side.
- At least 75 % of utterances have the expected sign.
- The placebo is ≈ 0: its CI includes 0, or |placebo median| ≤ 0.2 × |run median|.

How each measure scored:

- **dA** passes the first two criteria. It fails the third: the placebo is
  +0.037 with a CI excluding 0, and 0.037 > 0.2 × 0.078 = 0.016.
- **dV** passes the CI criterion but fails the share criterion (65 %) and the
  placebo criterion (+0.029, CI excluding 0).

Per-utterance n at k = 2: 400 (audio_chunks, video_chunks), 388 / 379
(both_chunks dA / dV). The k = 0 rows for both_chunks are in `summary.md`.

**Why the placebo is not ≈ 0 (POST HOC, exploratory; not pre-registered, not
part of the verdict).** `posthoc_chunk_placebo.py` reads the gated placebo
frames. Uniform chunk starts rarely cover the first and last ~10 frames of an
utterance, which are mostly leading and trailing silence, where the clean gate
weight is low (Part A).

| distance from the nearest utterance edge | 0–4 frames | 5–9 | 10–24 | ≥ 25 |
|---|---:|---:|---:|---:|
| median w (gated placebo = clean input) | 0.409 | 0.398 | 0.478 | 0.581 |
| P(frame inside any mask) | 0.094 | 0.209 | 0.364 | 0.404 |

The uncorrupted ("neither") frames are therefore enriched with low-weight
silence, and w(chunk) − w(neither) > 0 without any corruption. Restricting to
frames ≥ 25 from both edges (four-category reading, k = 0) shrinks the
placebo:

- dA: +0.045 → +0.011.
- dV: +0.027 → +0.006.

The corrupted runs then read dA −0.111 (audio_chunks) and dV +0.038
(video_chunks). The residual placebo offset (~+0.01) is unexplained; pauses
inside the utterance are a plausible but untested cause.

**Descriptive readouts (pre-registered outputs, not the decision rule).**

1. **Chunk-locked response** (`gate_chunk_locked.png`; mean w(condition) −
   w(placebo) on the same frames, aligned to the chunk onset or offset).
   - Audio chunks:
     - The weight starts falling ~2 frames before onset (−0.025 at t = −2).
     - It reaches −0.105 by frame +3 and holds at −0.10 to −0.12 through the chunk.
     - It recovers within ~3 frames after the offset.
     - A residual of ≈ −0.02 remains outside the chunks: an utterance-wide shift on top of the local response.
   - Video chunks: the weight rises to a plateau of +0.03 over ~5 frames, with a residual of ≈ +0.003 outside.
   - The 1–3-frame spill at each edge matches the encoders' context mixing. The pre-registered k = 2 exclusion is about the right width.
2. **The audio response is ~4× the video response** (corrected −0.115 vs
   +0.027). The two corruptions are not matched in severity: chunk corruption
   costs ASR +30 pt (3.87 → 34.00) and VSR +10 pt (26.36 → 36.50). The ratio
   says nothing about the gate's relative sensitivity at equal damage.
3. **Both streams corrupted** (both_chunks, frames in both masks, k = 2): the
   both − neither difference is −0.048 [−0.058, −0.039] (placebo +0.071). The
   gate moves toward video, so the audio corruption dominates.
4. **By video type** (video_chunks, k = 2): blur +0.084 [+0.055, +0.112]
   (placebo +0.039), occlusion +0.054 [+0.026, +0.072] (placebo +0.037).
5. **Traces** (`gate_chunk_traces.png`, the 8 §2.3 trace utterances).
   - In speech, the response appears as a step inside the chunk.
   - In pauses (w ≈ 0.35–0.45), neither corruption moves the weight visibly. Examples: the ROHAN4600_3191 video chunk at 3.8–4.5 s and the ROHAN4600_4174 audio chunk at 0.1–0.5 s.
   - This was seen in 8 utterances and not quantified. It is consistent with Part A.

**CER (%).** Δ in pt, with a paired bootstrap 95 % CI over utterances
(B = 10,000, one resample shared by all rows). ASR and VSR each ran under
the same corruption of their own stream; the condition used is in
parentheses. "Best single" = the better of those two.

| condition | AVSR static | AVSR gated | gated − static [CI] | gated better / worse / tie | ASR | VSR | static − best single [CI] | gated − best single [CI] |
|---|---:|---:|---|---|---:|---:|---|---|
| placebo (clean) | 3.13 | 3.52 | +0.39 [+0.10, +0.69] | 61 / 89 / 250 | 3.87 (placebo) | 26.36 (placebo) | −0.74 [−1.14, −0.36] vs ASR | −0.35 [−0.80, +0.08] |
| audio_chunks | **11.69** | 12.57 | +0.88 [+0.33, +1.45] | 123 / 161 / 116 | 34.00 (audio_chunks) | 26.36 (placebo) | −14.67 [−15.76, −13.60] vs VSR | −13.79 [−14.89, −12.69] |
| video_chunks | 3.60 | 4.00 | +0.41 [+0.07, +0.75] | 67 / 92 / 241 | 3.87 (placebo) | 36.50 (video_chunks) | −0.28 [−0.68, +0.11] vs ASR | +0.13 [−0.33, +0.59] |
| both_chunks | **13.60** | 14.55 | +0.95 [+0.35, +1.57] | 124 / 167 / 109 | 34.00 (audio_chunks) | 36.50 (video_chunks) | −20.40 [−21.69, −19.12] vs ASR | −19.44 [−20.77, −18.15] |

Bold = better arm with |Δ| ≥ 0.5 pt and a CI excluding 0. The placebo row
reproduces the §2.3 clean result bit for bit. Eval logs
(`auto_avsr/lightning_logs/`):

- AVSR static: version_70 / 72 / 74 / 76 (placebo, audio, video, both).
- AVSR gated: version_71 / 73 / 75 / 77.
- ASR: version_78 / 79 (placebo, audio).
- VSR: version_80 / 81 (placebo, video).

**Findings (Part B CER).**

1. **Under chunk-wise audio corruption, the gated arm is worse.** Chunk-wise
   corruption is the non-stationary case that per-frame gating was meant to
   exploit (§2, future work).
   - Static is better by +0.88 pt [+0.33, +1.45] (audio_chunks) and +0.95 pt [+0.35, +1.57] (both_chunks). Both gaps exceed 0.5 pt with CIs excluding 0.
   - The video_chunks (+0.41) and clean (+0.39) gaps are below 0.5 pt and are not called, even though their CIs exclude 0 (CLAUDE.md §8).
   - The intervals cover test-utterance sampling only. There is one training seed per arm, and neither arm saw chunk-wise corruption in training (babble, one SNR per utterance).
2. **No arm is worse than the better single modality in any condition.**
   - Fusion beats it by 14–20 pt under audio and both-stream corruption.
   - Under video corruption, both arms tie clean ASR (CIs span 0).
3. **Video chunks cost fusion ~0.5 pt** (static 3.13 → 3.60, gated 3.52 →
   4.00), although they cost VSR 10 pt. **Audio chunks cost fusion
   8.6–9.1 pt** (static 3.13 → 11.69, gated 3.52 → 12.57).

**Scope of claims.**

- **What this test supports:** an utterance-level reliability interpretation of the gate (§2.3).
- **What it does not support:** a per-frame one.
- **Why it failed:** the control failed, and for video so did the consistency criterion. The per-frame weight does change locally where corruption is applied (descriptive readouts 1–5), but the pre-registered test cannot certify that. Per CLAUDE.md §5.2 it is not re-scored with a different rule.
- **What happens next:** CLAUDE.md requires the test to be repeated unchanged after chunk-wise training augmentation, and this result is that run's pre-augmentation baseline.
- **Open caveat for the repeat:** the placebo offset comes from mask placement against the clean gate's speech/pause pattern, so it is expected to recur in the repeat regardless of what the gate learns. Whether to amend the rule before that run is an open decision. Nothing has been changed.

**Divergences from CLAUDE.md §5.2 (all recorded in the plan entry before
the runs).**

- ASR/VSR placebo runs were added, as clean per-utterance single-modality references.
- The dirty-tree check ignores untracked eval logs under `auto_avsr/lightning_logs/`, because each run writes one.
- ASR receives raw-length audio, while AVSR audio is cut to 640 × video frames before the transform. The seeded `AddNoise` crop offset and reference power depend on length, so ASR gets the same noise file, SNR rule and chunk spans but a different noise realisation inside the chunks. This is a near-identical input, not a bit-identical one.
- "Holds clearly" and "placebo ≈ 0" were made operational as the rule above.
- The occluder split is by image, not by category: the two halves share COCO categories.

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
| Chunk sweep, AVSR static (placebo, audio, video, both) | version_41 | version_70, 72, 74, 76 |
| Chunk sweep, AVSR gated (placebo, audio, video, both) | version_42 | version_71, 73, 75, 77 |
| Chunk sweep, ASR (placebo, audio) / VSR (placebo, video) | (2026-07-08 run) / version_31 | version_78, 79 / version_80, 81 |

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
