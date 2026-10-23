# CLAUDE.md — gr_avsr research rules

This repo is a master's research project: Japanese audio-visual speech recognition (AVSR)
on the ROHAN4600 corpus (single RTX 4090, 24 GB), built on vendored Auto-AVSR and
reproducing Kondo & Tamura (APSIPA ASC 2025), framework (c). The rules below protect the
validity of the experiments. Treat them as hard constraints. If a task would break one,
stop and say so instead of working around it.

## 1. What the contribution is (do not expand it on your own)

- The contribution is the **combination** of four components applied together to
  low-resource Japanese AVSR: (a) audio input augmentation and (b) visual input
  augmentation (section 2), (c) per-frame dynamic fusion (`ModalityGate`, below), and
  (d) pseudo-labelling (section 6). None is claimed as new on its own; the claim is that
  this combination has not been reported for Japanese.
- Because the claim is a combination, **every component must be separable**. Implement each
  behind its own config flag, off by default, so that any subset can be trained and each
  component can be ablated on its own. Never couple two components in one flag or one code
  path. Defaults must reproduce the current runs exactly.
- Every component is kept regardless of whether it improves CER. A component that shows no
  gain is reported as such; do not remove, retune or quietly redesign it to make it help.

Rules for the gate:

- `ModalityGate` (`espnet/nets/pytorch_backend/nets_utils.py`) is used in `E2E.fuse`
  (`espnet/nets/pytorch_backend/e2e_asr_conformer_av.py`). It is a small MLP that outputs
  per-frame softmax weights (visual, audio). Each stream is scaled by `2 * weight` before
  the existing `MLPHead` fusion.
- It is trained from task loss only. Never add SNR labels, reliability targets, auxiliary
  losses, or SNR as an input feature unless explicitly asked. That would change what the
  contribution is.
- The final gate layer is zero-initialised, so the untrained gate is exactly the static
  concatenation baseline. Preserve this property in any change to the gate or `fuse`.
- Do not add capacity or complexity to make the method look more novel. In this low-data
  regime the restricted gate is intentional. Propose larger architectural changes in
  chat; do not implement them on your own.

## 2. Ablation integrity (the most important section)

- **Equal treatment.** Every arm being compared (ASR, VSR, AVSR static, AVSR gated, and any
  future variant) must use the same noise-augmentation regime, hyperparameters, sample
  rate, data splits, epochs, checkpoint averaging and evaluation sets. The only allowed
  difference is the one being studied.
- **Static baseline = frozen gate.** The static arm is the same model with
  `model.audiovisual_backbone.fusion_gate_freeze=true` (see
  `configs/experiment/rohan_avsr_fixed.yaml`). Keep it that way. Do not fork a separate
  static-fusion code path, because that would break the identical-except-one-flag twin
  design.
- **Noise augmentation is load-bearing.** The gate learns reliability only from SNR
  variation in training. Never turn off or narrow training noise (`AddNoise`,
  `data.noise_snr_levels`, `data.noise_prob`) for the gated arm alone. A gated run trained
  without SNR variation is invalid.
- **Input augmentation (audio and visual).** The scheme follows AV-RelScore-style
  chunk-wise corruption (Hong et al., "Watch or Listen", CVPR 2023): within each utterance, random
  segments of the audio and/or video stream are corrupted, so the two streams can be
  degraded at the same time or alternately. Training-time augmentation is random; any
  evaluation-time corruption must be seeded and identical across arms. When augmentation is
  not the factor under study, every compared arm trains with the same augmentation.
- **Keep train and eval corruption disjoint.** Noise types, occluder images and corruption
  settings used for evaluation must not be used in training. Record which is which in the
  config, not only in code.
- **Check before you report.** Before reporting any comparison, confirm that both arms'
  `hparams.yaml` files agree on everything except the studied flag. List any differences.

## 3. Hyperparameters

- Use the inherited Kondo & Tamura recipe unchanged: AdamW, cosine schedule, 5-epoch
  warmup, peak LR 1e-4, CTC weight 0.1, front-ends as configured.
- Do not tune or search hyperparameters, even if it would improve results. "Inherits the
  prior recipe unchanged" is part of the paper's argument. Flag a suspected bad value in
  chat instead of changing it.
- Audio is 16 kHz with `rate_ratio` 640. The paper's 19.2 kHz / 768 setting is a documented,
  accepted deviation. Do not reintroduce it except to reproduce an old run explicitly.

## 4. Data and evaluation boundaries

- Test splits are never used for training, model selection, early stopping or tuning.
- CENSREC-1-AV is held-out out-of-distribution evaluation only. It must never enter
  training, validation, augmentation or any normalisation statistics.
- Training noise and evaluation noise must be disjoint by type. Currently, training uses
  `datamodule/babble_noise.wav` and evaluation uses NoiseX-92 white. Do not train on any
  noise type that is used for evaluation.
- SNR ladder for sweeps: clean, 20, 15, 10, 5, 0, −5, −10 dB, identical for all arms.
- Metric is CER (character-level, Katakana, `data.lang=ja`). Never label it WER.

## 5. Gate diagnostics (Phase 2c)

- The gate is "readable" by construction, but it may be called interpretable as
  reliability only once the diagnostic is validated. Do not write code comments, logs
  or reports that claim interpretability before that.
- The diagnostic must be computed on a frozen, final checkpoint. Never use the
  diagnostic curve as a training, tuning or checkpoint-selection target.
- Gate outputs are diagnostic only; do not feed them back into training.

### 5.1 Done 2026-10-23 (RESULTS.md §2.3): per-utterance gate test (inference only)

The existing sweep CSVs (`exp/snr_sweep_*.csv`) hold only one test-set mean per SNR
level, because `lightning_av.py` accumulates gate weights over the whole test set. The
per-utterance test needs a new inference-only sweep that logs per utterance. Rules:

- **No training, no model changes.** Use the existing `model_avg_10.pth` checkpoints of
  both twin arms (gated and static/frozen-gate) unchanged. Do not modify `ModalityGate`,
  `E2E.fuse` or the decoding settings. Logging must not change the aggregate CER or
  `gate_w_audio` that the current code reports.
- **Deterministic, identical noise.** Before running, make evaluation noise seeded and
  reproducible per utterance (the noise crop offset in `AddNoise` is currently unseeded).
  For each SNR level, every arm must see exactly the same noisy input for each utterance.
  Same NoiseX-92 white file, same SNR ladder (clean, 20, 15, 10, 5, 0, −5, −10 dB).
- **What to log, one row per (arm, SNR level, utterance):** utterance ID, `snr_target`,
  number of frames, mean audio gate weight over that utterance's frames, character edit
  distance and reference length. Per-utterance edit counts, not just CER, are needed for
  paired bootstrap between arms. Write to a new file per run; do not append to or
  overwrite the existing sweep CSVs.
- **Per-frame traces.** Also save the full per-frame audio-weight trace for a small fixed
  set of utterances (chosen once, before looking at results) at every SNR level, so we
  can check whether the gate varies within an utterance beyond its mean.
- **Sanity check.** The static (frozen-gate) arm must read exactly 0.5 at every frame and
  utterance. If it doesn't, stop and report; something is wrong with the checkpoint or
  loading.
- **Scope of claims.** This test is the weak form of Phase 2c: utterance-level gate
  weight against the target SNR. It cannot validate the per-frame claim on its own, because
  uniform white noise at a fixed SNR gives the gate little within-utterance reason to vary.
  Report it as such. The strong form (per-frame weight against segmental SNR or a known
  per-frame corruption mask) is a separate, later test.

### 5.2 Current task: strong-form gate test (inference only)

§5.1 showed that the utterance-mean audio weight is ordered by target SNR. It did not
show that the per-frame weight tracks per-frame reliability, because uniform white noise
at a fixed SNR gives the gate no within-utterance reason to vary. This task tests that,
in two parts. Part A uses only existing outputs. Part B needs new evaluation runs.

General rules (all of §5.1's rules still apply):

- **No training, no model changes.** Use the same two `model_avg_10.pth` checkpoints as
  §5.1 (static sha256 `f543646f…`, gated `6f0291cb…`) unchanged. Do not modify
  `ModalityGate`, `E2E.fuse` or the decoding settings.
- **Pre-register.** Fix every corruption setting, mask rule, threshold and analysis
  choice in config and code, and commit them **before** the first real run. Smoke tests
  run from the scratchpad, not in `lightning_logs`. Real runs must log a clean git tree
  (`git.dirty == false` in the meta file); abort if it is dirty.
- **Never tune toward the result.** If the gate fails this test, report it as failing.
  Do not change chunk settings, thresholds or corruption types afterwards to make it
  pass. A failure here is the pre-augmentation baseline for the same test after
  chunk-wise training augmentation, not a bug.
- **Frame alignment.** One frame = 40 ms = 640 audio samples at 16 kHz = one video frame
  at 25 fps. Assert that the gate trace length, the number of video frames and
  `audio_samples / 640` agree for every utterance. Stop and report if they don't.

#### Part A: speech activity and segmental SNR on the existing traces (no GPU)

- **Inputs.** Use the eight trace utterances in `decode.trace_utterances` and the
  `*_traces.json` files in `exp/per_utt_sweep/`. Rebuild each utterance's noise crop
  from (`noise_seed=0`, utterance ID) with the same code path as `AddNoise`, and check
  it against the logged `audio_sha1`.
- **Per-frame references, computed from the clean audio.** Speech activity uses an
  energy-based VAD with a threshold fixed before looking at any trace. Segmental SNR is
  10·log10(speech energy / noise energy) within each 640-sample frame, at each SNR
  level.
- **Report.** Give the mean gate weight in speech frames vs non-speech frames per
  utterance and level, and the per-utterance Spearman correlation between per-frame
  weight and segmental SNR.
- **Scope.** Under stationary white noise, segmental SNR is almost entirely driven by
  speech energy, so it is confounded with speech activity. Part A can show whether the
  stretches converging to ≈ 0.40–0.45 (RESULTS.md §2.3, finding 3) are pauses. It
  cannot separate reliability from speech activity; Part B does that.
- **Outputs.** Write new files only (e.g. `exp/per_utt_sweep/partA_*`). Never modify the
  §5.1 outputs.

#### Part B: chunk-wise corruption with a known per-frame mask

**Harness.**

- Add seeded, eval-only chunk corruption to the test transforms, behind new `decode.*`
  flags that are off by default. With the flags off, the train/val/test pipelines must be
  bit-identical to HEAD, including RNG consumption. Regression-test this as was done for
  `noise_seed`.
- All draws are a function of (`noise_seed`, utterance ID, stream), so every arm sees the
  identical corrupted input. Verify this with input hashes for both audio and video.
- Each transform returns its per-frame boolean mask, which is logged alongside the gate.
- Corruption is applied to the model's actual inputs: audio before
  `_instance_layer_norm`, video on the preprocessed mouth ROI in the `VideoTransform`
  test path.

**Chunk settings.** These are defaults to confirm or change, then fix before running.

- Per stream, independently: 1–3 chunks per utterance, each 10–25 frames (0.4–1.0 s),
  with uniform random placement.
- Sampling audio and video independently means chunks overlap in some utterances and
  alternate in others, so all four reliability combinations occur.

**Corruption types.** Train/eval disjointness is binding (§2 and §4).

- **Audio:** NOISEX-92 white (`white_16k.wav`, already eval-only), added inside the
  chunks only, at a fixed per-chunk SNR of −5 dB. Audio outside the chunks stays clean.
- **Video:** choose the evaluation types now and record them in the config. Once chosen,
  they are **excluded from all future training augmentation**. Suggested: occlusion with
  patches from a held-out occluder image set, plus Gaussian blur. Training visual
  augmentation will then use the remaining AV-RelScore-style types and a disjoint
  occluder set.

**Conditions** (one run per arm × condition):

- `audio_chunks`: audio corrupted in chunks, video clean.
- `video_chunks`: video corrupted in chunks, audio clean.
- `both_chunks`: both streams corrupted, with masks drawn independently.
- `placebo`: the same masks as `both_chunks` are drawn and logged, but no corruption is
  applied. This is the control. Gate differences inside vs outside the masks must be
  ≈ 0 here; if they are not, the analysis is picking up something other than corruption.

**Arms.**

- AVSR static (`fusion_gate_freeze=true`) and AVSR gated (`false`). The static arm must
  still read exactly 0.5 everywhere; stop and report if it doesn't.
- ASR under `audio_chunks` and VSR under `video_chunks`, with the same seeds, as
  single-modality references for the "no worse than the better single modality" check.

**Logging.**

- One row per (arm, condition, utterance): utterance ID, `n_frames`, fraction of frames
  corrupted per stream, mean audio gate weight inside and outside each mask (audio-only
  corrupted, video-only corrupted, both, neither), character edit distance, reference
  length, and audio and video input hashes.
- Per-frame gate weights and both masks for **all** test utterances, one compressed
  `.npz` per run. This is small, about 72k frames per run.
- Write each run to a new directory, `exp/chunk_sweep/<exp_name>/<condition>`, and refuse
  to overwrite.

**Analysis.** The script is written and committed before any real run.

- **Primary measure, per utterance (paired within the utterance):**
  - Mean audio weight in audio-corrupted frames minus mean in uncorrupted frames.
    Expected sign: negative.
  - Mean audio weight in video-corrupted frames minus mean in uncorrupted frames.
    Expected sign: positive.
- Report the median and 95 % bootstrap CI across utterances, the fraction of utterances
  with the expected sign, and the frame-level AUC. Put the placebo result beside every
  number.
- **Boundary smearing.** The Conformer encoders mix context across frames, so the gate's
  response may spread past chunk edges. Report every measure twice: with all frames, and
  with k = 2 frames excluded on each side of every chunk boundary. Fix k before running.
- **CER.** Report the twin CER gap per condition with a paired bootstrap, using the §8
  convention: a gap below ~0.5 pt is not called a win, even if its CI excludes 0. Also
  compare against the ASR/VSR references.

**Scope of claims.**

- If the expected signs hold clearly and the placebo is ≈ 0, the per-frame weight tracks
  the corrupted stream for these corruption types only. That is the evidence needed to
  call the gate interpretable as reliability, restricted to the tested conditions.
- If they do not hold, report the gate as reliability-sensitive at the utterance level
  only (§5.1). This test is then repeated unchanged after chunk-wise training
  augmentation.

---

Related edits elsewhere in CLAUDE.md:

- §5.1 heading: change "Current task" to "Done 2026-10-23 (RESULTS.md §2.3)".
- §8, after the 0.5 pt rule, add: "When a paired-bootstrap CI is available, report it,
  but a gap below ~0.5 pt is still not called a win even if its CI excludes 0: the CI
  covers test-utterance sampling only, not training-seed variance."


## 6. Pseudo-labelling

- Pseudo-labelling (ASR transcripts converted to Katakana) is part of the system and is
  implemented regardless of how the gate performs.
- Keep pseudo-labelled data physically separate from ROHAN: its own dataset directory and
  label CSVs, enabled by its own config flag. It never enters ROHAN val/test, CENSREC-1-AV,
  or any evaluation set, and test transcripts are never used to generate or filter labels.
- Always report results both with and without pseudo-labels. The manual-transcription-only
  number must remain available for every configuration that is reported.
- Record provenance in the dataset directory: source audio/video, ASR model and version,
  Katakana conversion tool, and any confidence filtering.
- Every compared arm receives the same pseudo-labelled data.
- AVSpeech and VoxCeleb2 are unavailable; do not build pipelines that depend on them.

## 7. External baselines

- AVista (Reazon/Enactic) is an external baseline only. Evaluate its public weights
  through this repo's evaluation harness. Do not rebuild this project on its stack.

## 8. How to report work

- Separate what is true by architecture from what still needs empirical validation.
- Whenever the implementation diverges from the specification in this file or in
  `REPRODUCTION_PLAN.md`, say so explicitly, even if the divergence seems harmless.
- Report numbers with the checkpoint, config overrides, noise file and SNR used.
- Gaps under ~0.5 pt CER are within noise for this setup. Do not call them wins.
  When a paired-bootstrap CI is available, report it, but a gap below ~0.5 pt is still
  not called a win even if its CI excludes 0: the CI covers test-utterance sampling only,
  not training-seed variance.
- Keep `RESULTS.md` and `REPRODUCTION_PLAN.md` the sources of truth. Update them when
  results or commands change, and mark superseded content instead of silently deleting it.

## 9. Operational rules (from the incident log)

- **Shell is Windows PowerShell 5.1.** No bash syntax: no `&&`, no `\` continuation (use a
  trailing backtick), no `VAR=x cmd`. Run from `auto_avsr\` with the venv active and
  `$env:PYTHONIOENCODING = "utf-8"` set.
- **Data identity lives in experiment configs.** The default `configs/data/dataset/rohan.yaml`
  points at the broken `datasets/rohan_preprocessed`. Always use a config that bakes in
  its root (`rohan_avsr_fixed`, `rohan_vsr_fixed`, ...). Never rely on a CLI override for
  which data a run sees.
- **Verify runs from logged `hparams.yaml`,** not from the command you think you ran:
  dataset root, checkpoint, `snr_target`, `noise_filename`.
- Training logs go to `auto_avsr/exp/lightning_logs/version_N`, evals to
  `auto_avsr/lightning_logs/version_N`. Version numbers overlap; always name the dir.
- Logged `cer` is a fraction (0.1048 = 10.48 %).
- `eval_sweep.py` appends to `exp/snr_sweep_<exp_name>.csv`. Rename existing CSVs before a
  new sweep.
- **No mid-run parameter changes.** Crash recovery = append `ckpt_path=exp/<exp_name>/last.ckpt`
  and change nothing else.
- VRAM: AVSR at fp32 needs `max_frames` ≤ 800 (1000 spills to system RAM under WDDM).
  `num_workers=2` is the tested setting.
- `auto_avsr_original_do_not_touch/` is read-only.

## 10. Documentation map (read on demand, do not merge into this file)

- `RESULTS.md` — current numbers and their provenance. Source of truth for results.
- `REPRODUCTION_PLAN.md` — chronological experimental log with exact commands. Earlier
  entries contain plans and "pending" notes that later entries resolve: read each item to
  its end before acting on it, and never execute an old plan step without checking it is
  still open.
- `SYSTEM_STATUS.md` — environment and code modifications (§2–§5 valid). Its §1 dataset path
  and 4.2 % milestone are obsolete.
- `auto_avsr/PIPELINE.md`, `auto_avsr/JAPANESE_AVSR.md` — legacy. Their commands use the
  broken dataset and old front-ends; do not run them.
