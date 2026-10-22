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

### 5.1 Current task: per-utterance gate test (inference only)

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
