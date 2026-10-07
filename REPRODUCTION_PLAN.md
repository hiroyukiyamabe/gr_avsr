# Plan: Clean Tamura Reproduction (3,448 h frontends) + Dynamic-Fusion Swap

> **Final results tables: see `RESULTS.md`** (complete reproduction table,
> Table IV replica, twin-study verdict, VSR ablations, run provenance,
> documented deviations). This file remains the experimental log.

**Goal.** Reproduce Kondo & Tamura (APSIPA ASC 2025) framework (c) — cross-lingual
transfer, all three modalities — on our own ROHAN recordings, using front-ends
extracted from the canonical 3,448 h Auto-AVSR checkpoint. Then compare static
fusion (their model) against our dynamic per-frame fusion gate via the twin-run
workflow, and replicate their noise table (Table IV) with white noise.

**Key design decisions (settled 2026-07-07):**

- Frontend source: `avsr_trlrwlrs2lrs3vox2avsp_base.pth` from the auto_avsr-1.0.0
  model zoo (the paper's "original English model", Fig. 1, 0.93% WER). Both the
  visual and audio front-ends are extracted from this **single** AV checkpoint so
  every modality comparison shares one source. Fallback ablation if VSR still
  misses target: modality-matched `vsr_trlrwlrs2lrs3vox2avsp` / `asr_trlrwlrs2lrs3vox2avsp`.
- Static-fusion arm = gated code path with `fusion_gate_freeze=true` (verified
  numerically identical to the original concat-MLP fusion). Dynamic arm = same
  command with `fusion_gate_freeze=false`. No fusion code changes needed.
- Reproduction metric checkpoint: `model_avg_10.pth` (last-10 average — the
  original Auto-AVSR recipe). Export best-val too, as a diagnostic only.
- Dataset: `datasets/rohan_preprocessed_fixed` (25 fps, 96×96 central mouth ROI,
  16 kHz audio, 3800/400/400 split).

---

## Paper targets (framework (c), Tables III–IV)

| Experiment | Paper CER | Notes |
|---|---|---|
| ASR (16 kHz) | 3.88% | paper's best ASR (19.2 kHz) is 3.34% |
| VSR (25 fps) | 19.70% | previous best (ours): 27.26% w/ 438 h frontend |
| AVSR clean | 2.78% | old 4.2% run used broken video — not comparable |
| AVSR @ SNR 10 dB (white) | 11.95% | ASR 16 kHz @ 10 dB: 26.02% |
| AVSR @ SNR 5 dB (white) | 16.11% | ASR 16 kHz @ 5 dB: 40.13% |

## Accepted deviations from the paper (document, don't fix)

> **2026-07-11 CORRECTIONS (from re-reading both papers — APSIPA 2025 +
> SIGUL 2024):** deviations #1 and #3 below were based on WRONG assumptions.
> (1) The paper used a PUBLIC ROHAN multimodal recording (SIGUL: 300x300
> mouth videos, "4,200 available to the public as of Feb 2024") — NOT their
> own recordings. **But which speaker is UNRESOLVED**: the zunko.jp
> multimodal database hosts ROHAN4600 recordings by MULTIPLE characters
> (Itako/Zundamon/Metan/Sora/Usagi) in the same format, and the duration
> fingerprint conflicts: their 7.7 h / 3,400 train sentences = 8.15 s/utt vs
> our zundamon 6.70 s/utt (3,400-equiv = 6.33 h). Two hypotheses: (a) a
> DIFFERENT, ~22%-slower speaker (slower speech = more frames/mora = easier
> VSR — a live gap explanation); (b) same zundamon data with duration booked
> at 25 fps on ~30 fps frame counts (6.33 x 1.2 = 7.6 ~ "7.7 h" and matches
> their 3,448/450 arithmetic) — suspicious coincidence but not proof.
> Discriminating test: zunko.jp registration (per-speaker durations/release
> counts) or asking the authors.
> (3) Their ROI is NOT landmark-tracked: SIGUL §3.3 states the video is
> resized from 300x300 to 96x96 WHOLE-FRAME (no mouth crop at all), 25 fps,
> grayscale, Normalize(0.421, 0.165). Their mouth is therefore ~31px wide
> (~0.69x the auto_avsr canonical 45px) with the full lower face in frame,
> vs our mouth-centred ~45-53px crops. **This exact preprocessing difference
> is the new prime suspect for the VSR gap** (same corpus, same recipe
> otherwise). Also confirmed: 87 katakana IDs (ours matches), AdamW/1e-4/
> warmup 5/weight decay 0.03/60 ep/max_frames 1600, single RTX 3090.

1. ~~Own ROHAN4600 recordings~~ CORRECTED: same public corpus; remaining
   data deviations are only the split (we train 3,800 of 4,600; they trained
   3,400 of the 4,200 available in Feb 2024) and different test items.
2. ~~Audio 16 kHz + `rate_ratio=640` vs their 19,200 Hz + ratio 768~~
   REVISED 2026-07-11: 16 kHz stays for ASR ONLY (the paper reports an ASR row
   at 16 kHz — our 3.87% matches it). The AVSR arms will MATCH the paper's
   19,200 Hz: every AVSR number in the paper (clean 2.78% and all of Table IV)
   exists only at 19.2 kHz, and Table IV shows the rate interacts with noise
   robustness non-trivially (ASR@19.2k degrades worse in noise than @16k), so
   a 16 kHz twin study would be incommensurable with their noise table. Raw
   corpus audio is 96 kHz; both rates are exact decimations (÷5 vs ÷6) with
   exact per-frame ratios at 25 fps (768 vs 640). See "Pre-AVSR audio switch"
   in the checklist.
3. ~~Central 200→96 mouth crop vs their landmark-tracked 96×96 ROI~~
   CORRECTED: their ROI = whole-frame 300→96 resize (SIGUL §3.3), no crop, no
   landmarks. Our mouth-centred crops (200px and 234px variants) are the
   deviation; the whole-frame variant is untested and is the reproduction
   candidate ("paper-prep" run).
4. VSR `max_frames=1000`, `num_workers=2` vs paper's 1600 (hardware stability:
   1600/4-workers crashed with cuDNN/fragmentation; 1000/2 ran flat ~10.5 min/epoch).
   ASR keeps the paper's 1600/60. AVSR's 1000/40 matches the paper exactly.
5. **No mid-run parameter changes this time** — the 1600→1000 mid-run switch broke
   the cosine schedule in the previous VSR run and is a suspected contributor to
   the 27.26 vs 19.70 gap.

---

## Code changes (4 items, all in this repo)

1. **`module_extractor/extractor.py` — AV-checkpoint mode.** Currently only strips
   the single-modality `frontend.` prefix. Add `--av`: read the AVSR checkpoint,
   write `encoder.frontend.*` → `frontends/frontend_visual_3448.pth` and
   `aux_encoder.frontend.*` → `frontends/frontend_audio_3448.pth` (module names
   confirmed in `espnet/nets/pytorch_backend/e2e_asr_conformer_av.py`). Output
   format unchanged (`trunk.*`/`frontend3D.*` wrapped in `model_state_dict`), so
   `lightning.py`/`lightning_av.py` load them as-is.

2. **`auto_avsr/eval.py` — restore the original.** Rename current file to
   `eval_sweep.py` (keeps the SNR-sweep CSV + `plot_snr_sweep.py` workflow for the
   fusion study). New `eval.py` = faithful port of auto_avsr-1.0.0 `eval.py`:
   strict `load_state_dict`, plain `trainer.test`, no CSV. Only permitted
   deviations: `Trainer(accelerator="auto", devices=1)` (PL 2.6 API) and the
   `"video"` modality name. Strict loading is safe: all new checkpoints are trained
   through the gated code path and therefore contain `fusion_gate.*` keys.

3. **White-noise support (Table IV).** Add `decode.noise_filename` config key
   (default = existing babble file, so training behavior is untouched) and thread
   it through the datamodule into the eval-time `AddNoise(...)`
   (`datamodule/transforms.py:174`). Prepare the noise file: `AddNoise` asserts
   16 kHz; NoiseX-92 is natively 19.98 kHz — check
   `noise_datasets/NoiseX-92/white.wav` and resample once to
   `noise_datasets/white_16k.wav` if needed.

4. **New config `configs/experiment/rohan_avsr_fixed.yaml`.** Clone of
   `rohan_avsr` (40 epochs / `max_frames=1000` already match the paper) with
   `root_dir=…/rohan_preprocessed_fixed` and the 3,448 h frontend paths documented
   in the header.

---

## Commands

**Shell: Windows PowerShell 5.1.** Do NOT use bash syntax: no `&&` (run commands
on separate lines or join with `;`), no `\` line continuation (PowerShell uses a
trailing backtick `` ` ``), no `VAR=x cmd` env prefix, no `for x in …; do`.
All commands below are PowerShell-ready. Long commands use backtick
continuation — the backtick must be the LAST character on the line (no trailing
spaces), or paste the command as a single line.

Setup for every session (activate the project venv, go to the training dir):

```powershell
cd C:\Users\Hiroyuki\gr_avsr
& .\avsr\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
```

### Phase 0 — assets (once, from repo root `C:\Users\Hiroyuki\gr_avsr`)

```powershell
# Download the canonical AV checkpoint (Drive link from auto_avsr-1.0.0 README;
# MD5 must start with 6b3c5)
pip install gdown
gdown 1mU6MHzXMiq1m6GI-8gqT2zc2bdStuBXu -O original_checkpoints\avsr_trlrwlrs2lrs3vox2avsp_base.pth

# Verify the download (compare against 6b3c5…)
Get-FileHash -Algorithm MD5 original_checkpoints\avsr_trlrwlrs2lrs3vox2avsp_base.pth

# Extract both front-ends from the single AV checkpoint (after code change #1).
# extractor.py resolves ckpt/out dirs relative to its own location, so this
# works from the repo root:
python module_extractor\extractor.py --av --src avsr_trlrwlrs2lrs3vox2avsp_base.pth
# -> module_extractor\frontends\frontend_visual_3448.pth
# -> module_extractor\frontends\frontend_audio_3448.pth
```

### Phase 1 — fine-tunes (4 runs, from `auto_avsr\`)

```powershell
cd C:\Users\Hiroyuki\gr_avsr\auto_avsr

# 1) ASR — paper spec exactly (60 epochs, max_frames 1600). ~fast.
python train.py +experiment=rohan_asr exp_dir=exp exp_name=rohan_asr_3448 `
  data.dataset.root_dir=../datasets/rohan_preprocessed_fixed `
  pretrained_model_path=../module_extractor/frontends/frontend_audio_3448.pth `
  +data.num_workers=2

# 2) VSR — 60 epochs; 1000 frames / 2 workers (stability constraint). ~10.5 h.
python train.py +experiment=rohan_vsr_fixed exp_dir=exp exp_name=rohan_vsr_3448 `
  pretrained_model_path=../module_extractor/frontends/frontend_visual_3448.pth `
  data.max_frames=1000 data.max_frames_val=1000 +data.num_workers=2

# 2b) VSR frontend ablation — TRIGGERED 2026-07-09 (AV-frontend VSR = 26.36%,
#     above the ~24% threshold). Modality-matched VSR checkpoint from the
#     auto_avsr-1.0.0 zoo (Drive 19GA5SqDjAkI5S88Jt5neJRG-q5RUi5wi, MD5 a896f…,
#     3,448 h, 20.3% WER). Download + extract from the repo root:
#       gdown 19GA5SqDjAkI5S88Jt5neJRG-q5RUi5wi -O original_checkpoints\vsr_trlrwlrs2lrs3vox2avsp_base.pth
#       Get-FileHash -Algorithm MD5 original_checkpoints\vsr_trlrwlrs2lrs3vox2avsp_base.pth  # a896f…
#       python module_extractor\extractor.py --src vsr_trlrwlrs2lrs3vox2avsp_base.pth --dst frontend_visual_3448_vsr.pth
#     Then train (identical to run 2 except frontend + exp_name):
python train.py +experiment=rohan_vsr_fixed exp_dir=exp exp_name=rohan_vsr_3448_vsrfe `
  pretrained_model_path=../module_extractor/frontends/frontend_visual_3448_vsr.pth `
  data.max_frames=1000 data.max_frames_val=1000 +data.num_workers=2

# 3) AVSR arm A: STATIC fusion = Tamura reproduction (frozen gate). ~7 h.
#    (fixed root_dir + 3,448h frontend paths are baked into rohan_avsr_fixed.yaml)
python train.py +experiment=rohan_avsr_fixed exp_dir=exp exp_name=rohan_avsr_3448_static `
  model.audiovisual_backbone.fusion_gate_freeze=true +data.num_workers=2

# 4) AVSR arm B: DYNAMIC fusion — identical except the gate trains. ~7 h.
python train.py +experiment=rohan_avsr_fixed exp_dir=exp exp_name=rohan_avsr_3448_gated `
  model.audiovisual_backbone.fusion_gate_freeze=false +data.num_workers=2

# Crash recovery (any run): append  ckpt_path=exp/<exp_name>/last.ckpt
# and change NOTHING else (keep the cosine schedule intact).
```

### Phase 2 — clean evaluation (restored original eval.py; avg-10 checkpoint)

```powershell
cd C:\Users\Hiroyuki\gr_avsr\auto_avsr
$env:PYTHONIOENCODING = "utf-8"

# ASR
python eval.py +experiment=rohan_asr exp_dir=exp exp_name=rohan_asr_3448 `
  data.dataset.root_dir=../datasets/rohan_preprocessed_fixed `
  pretrained_model_path=exp/rohan_asr_3448/model_avg_10.pth transfer_frontend=false `
  +data.num_workers=2

# VSR
python eval.py +experiment=rohan_vsr_fixed exp_dir=exp exp_name=rohan_vsr_3448 `
  pretrained_model_path=exp/rohan_vsr_3448/model_avg_10.pth transfer_frontend=false `
  +data.num_workers=2

# AVSR (both arms; frontend paths/root_dir come from rohan_avsr_fixed.yaml)
python eval.py +experiment=rohan_avsr_fixed exp_dir=exp exp_name=rohan_avsr_3448_static `
  pretrained_model_path=exp/rohan_avsr_3448_static/model_avg_10.pth transfer_frontend=false `
  +data.num_workers=2
# repeat with exp_name=rohan_avsr_3448_gated

# NOTE: logged cer is a RAW FRACTION -> multiply by 100 for %.
# Diagnostic only (not the reproduction number):
#   python preparation\export_best_val.py exp\<exp_name>   # -> best_val_model.pth
```

### Phase 3 — Table IV replica (white noise; ASR + both AVSR arms)

```powershell
# after code change #3; noise file resampled to 16 kHz
foreach ($snr in 999999, 10, 5) {
  python eval.py <args as in Phase 2> `
    decode.snr_target=$snr `
    decode.noise_filename=../noise_datasets/white_16k.wav
}
# For the gated-vs-static SNR curves + gate-weight tracking, use eval_sweep.py
# (same args) — it appends to exp/snr_sweep_<exp_name>.csv for plot_snr_sweep.py.
```

---

## Order of execution / checklist

- [x] Code change #1 (extractor `--av`) — done 2026-07-08, tested on synthetic +
      real checkpoint; single-modality path regression-tested.
- [x] Code change #2 — done 2026-07-08. `eval_sweep.py` holds the sweep/CSV
      version; `eval.py` is the 1.0.0 original with only the two permitted
      deviations (PL 2.x Trainer API, "video" modality name). Strict load kept.
- [x] Code change #3 — done 2026-07-08. `decode.noise_filename` added (null =
      original babble; train-time noise untouched) and threaded through
      `data_module.py` → `AudioTransform("test", …)` → `AddNoise`.
      `noise_datasets/white_16k.wav` created (NoiseX-92 white, 19,980→16,000 Hz,
      235 s). Smoke-tested at SNR 10 dB.
- [x] Code change #4 — done 2026-07-08. `configs/experiment/rohan_avsr_fixed.yaml`
      with fixed root_dir + 3,448h frontend paths baked in; hydra composition
      verified (40 epochs, 1000 frames, gate_freeze override works, all paths exist).
- [x] Download AV checkpoint + extract front-ends — done 2026-07-08. MD5 verified
      (6B3C53AE…). `frontend_{visual,audio}_3448.pth` written, 120 tensors each,
      key/shape layout verified IDENTICAL to the known-good trlrs3 frontends
      (strict `transfer_frontend` load guaranteed).
- [x] Train ASR → eval — done 2026-07-08. **CER 3.87% vs paper 3.88%** (16 kHz,
      framework (c)) — exact reproduction. 60 epochs in ~1.5 h; best-val at
      epoch 58 (loss_val 4.160, late-epoch best = healthy, no overfitting);
      evaluated `model_avg_10.pth` via restored original eval.py.
- [x] Train VSR → eval (target ≈ 19.7%; if ≥ ~24%, run the modality-matched
      `vsr_trlrwlrs2lrs3vox2avsp` frontend ablation before touching the crop)
      - **Rerun (2026-07-09) VALID — CER 26.36%** (`model_avg_10.pth`, fixed
        dataset confirmed in eval log `auto_avsr/lightning_logs/version_10/hparams.yaml`).
        Better than the 438 h frontend (27.26%) but the AV-checkpoint frontend
        alone does NOT close the gap to 19.70% → the ≥ ~24% condition is
        **TRIGGERED**: run the modality-matched frontend ablation (step 2b below)
        before touching the crop.
      - **First attempt (2026-07-08) INVALID — 150% CER, not a frontend result.**
        The plan's original VSR command omitted `data.dataset.root_dir`, so the
        run trained AND evaluated on the broken `rohan_preprocessed` videos
        (old 300px/30fps). Training curve = the known memorization collapse
        (best loss_val 93.4 @ epoch 9, then rising — same as the old 123% run).
        Fix: fixed root_dir is now BAKED INTO `rohan_vsr_fixed.yaml`; commands
        above are correct as written. `exp/rohan_vsr_3448/` from the bad run
        (~33 GB of checkpoints) can be deleted before rerunning.
- [x] VSR frontend ablation (step 2b): `rohan_vsr_3448_vsrfe` with the
      modality-matched `vsr_trlrwlrs2lrs3vox2avsp` visual frontend. Eval as in
      Phase 2 with `exp_name=rohan_vsr_3448_vsrfe`. If this also lands ≥ ~24%,
      the frontend hypothesis is exhausted → next suspect is the central crop
      (accepted deviation #3).
      - **Result (2026-07-10): CER 26.63%** — statistically indistinguishable
        from the AV-checkpoint frontend (26.36%). Valid run (eval log
        `lightning_logs/version_11/hparams.yaml`: fixed dataset, right ckpt;
        ran with `num_workers=4`, a benign deviation from the plan's 2).
        **Conclusion: the frontend hypothesis is EXHAUSTED.** Neither frontend
        scale (438 h → 3,448 h: 27.26 → 26.36) nor modality-matched pretraining
        (26.63) moves VSR meaningfully. The remaining ~7-pt gap to the paper's
        19.70% now points at accepted deviation #3 (central 200→96 crop vs
        their landmark-tracked mouth ROI) and/or own-recording differences.
        Decision pending: fix the crop (re-preprocess + retrain all video runs)
        vs. accept ~26% VSR as a documented deviation and proceed to AVSR.
- [x] Crop audit (2026-07-10) — measured the fixed dataset's framing against the
      corpus's own MediaPipe landmark CSVs (478-pt FaceMesh, LFROI pixel coords;
      100 utterances, ~19k valid frames) and the auto_avsr canonical scale
      (20words_mean_face.npy: mouth width 45px in the 96px crop). **Verdict: the
      central crop is geometrically SOUND — the catastrophic-framing theory is
      dead.** Findings:
      - Containment: lips never leave the 96px crop or the CenterCrop(88)
        window (0 of 18,926 valid frames).
      - Position: mouth center extremely stable at (46.4, 40.0) vs target
        (48, 48) — a consistent ~8px-high offset; per-utterance mean-y std
        1.4px; rotation negligible (std 1.4°).
      - Scale: mouth width 52.8px vs canonical 45px = 1.17x zoom (modest).
      - Residual geometric delta (8px offset + 1.17x zoom) is small, systematic,
        and partially absorbed by fine-tuning + RandomCrop(88) augmentation;
        optional "recrop v2" = mouth-centered 234px box at LFROI (147,133) in
        normalize_rohan_video.py would close it exactly.
      - DATA-QUALITY NOTE: landmark CSVs for the ENTIRE 1201-1600 batch are
        degenerate (all points collapsed; mouth width 0). The videos themselves
        are fine (visually verified). Any future landmark-based preprocessing
        must interpolate/skip that batch.
      - Remaining explanations for the ~7pt VSR gap, in likely order: own
        recordings (speaker/style/quality), nearest-frame 30->25 fps resample
        judder, small geometry delta, paper-side unknowns.
- [ ] Recrop v2 VSR (decided 2026-07-10): close the audited geometry delta
      exactly and retrain once with the canonical AV frontend (same as the
      26.36% run, for a clean crop-only comparison).
      - `normalize_rohan_video.py` gained `--center-x/--center-y` (clamped
        static crop box; default = old centred behaviour). Smoke-tested
        visually: mouth lands on the canonical 45x20px extent, incl. the
        broken-landmark 1201-1600 batch.
      - Dataset: `datasets/rohan_preprocessed_recrop` via
        `--target-fps 25 --central 234 --center-x 147 --center-y 133
        --mouth-size 96` (launched 2026-07-10).
      - Config `rohan_vsr_recrop.yaml` (root_dir BAKED IN, hydra-verified).
      - Train (after normalization finishes), from auto_avsr\:
        `python train.py +experiment=rohan_vsr_recrop exp_dir=exp
        exp_name=rohan_vsr_3448_recrop
        pretrained_model_path=../module_extractor/frontends/frontend_visual_3448.pth
        data.max_frames=1000 data.max_frames_val=1000 +data.num_workers=2`
      - Eval as Phase 2 with `+experiment=rohan_vsr_recrop
        exp_name=rohan_vsr_3448_recrop`. Same epoch-10 loss_val<90 check.
      - Launched 2026-07-10 (training log version_33). Epoch-10 gate PASSED:
        loss_val 102 -> 32.0, monotone. Early comparison vs the 26.36% baseline
        (version_31): trajectories nearly identical, recrop ~1 epoch ahead
        (32.0 @ ep10 vs 33.0; baseline hit 31.9 at ep11) — consistent with a
        small geometry effect; final CER decides.
      - **Result (2026-07-11): CER 26.50%** (model_avg_10, eval log
        lightning_logs/version_12 verified: recrop data, right ckpt) vs
        baseline 26.36%. **GEOMETRY EXONERATED** — the 8px offset + 1.17x zoom
        have no measurable effect (fine-tuning absorbs them, as suspected).
        Per the pre-registered rule: AVSR arms proceed on
        rohan_preprocessed_fixed; the ~7pt VSR gap vs the paper's 19.70% is
        now attributed to own recordings (speaker/style/quality) and/or the
        30->25 fps nearest-frame resample, and is documented as a deviation.
        VSR reproduction line for the write-up: **26.36%** (canonical AV
        frontend, rohan_preprocessed_fixed), with 26.63% (modality-matched
        frontend) and 26.50% (recrop) as supporting ablations showing the
        number is robust to frontend choice and crop geometry.
      - Interpretation: vs 26.36% baseline. Big drop -> geometry mattered;
        AVSR arms then use the recrop data (clone rohan_avsr_fixed.yaml
        accordingly). No change -> geometry fully exonerated; AVSR arms run on
        rohan_preprocessed_fixed and the gap is attributed to recordings/fps.
      - Download + extraction done 2026-07-09 (MD5 a896f… verified). Extractor
        fix required: new-zoo single-modality checkpoints nest the front-end at
        `encoder.frontend.*` (not flat `frontend.*` like the trlrs3 files);
        `extract_frontend` now tries both prefixes (old layout regression-tested).
        `frontend_visual_3448_vsr.pth` verified: 120 tensors, key/shape layout
        identical to `frontend_visual_3448.pth`, weights differ (120/120) —
        strict `transfer_frontend` load guaranteed. Training not yet started.
- [ ] VSR "paper-prep" run (`rohan_vsr_3448_paper`): whole-frame 300→96 dataset
      (`rohan_preprocessed_paper`, normalize --central 300), paper
      max_frames=1600, VSR-checkpoint frontend (`frontend_visual_3448_vsr.pth`
      — SIGUL's source model). Config `rohan_vsr_paper.yaml` created +
      hydra-verified 2026-07-11. Read-out: ≈19.7% → framing was the gap, AVSR
      arms adopt this video prep; ≈26% → framing exonerated, residual gap =
      recording/speaker difference (speaker identity unresolved — see the
      2026-07-11 correction note above). Caveat: changes framing AND
      max_frames vs the 26.36% baseline; if it lands in between (~22–24%),
      one more ablation (paper-prep @1000) attributes the gain. Crash rule:
      1600 crash → fresh run at 1000/1000, no mid-run switches.
      - 2026-07-11: 1600 OOM'd on the RTX 4090 at fp32 (batch 4, epoch 0);
        expandable_segments retry ran but spilled to system RAM (<0.5 it/s,
        sysmem fallback) → aborted per plan. Fresh run at 1000/1000
        (version_36, ~1.8 it/s w/ 4 workers). NOTE: at 1000 frames the run is
        a single-variable framing twin of the 26.63% vsrfe run; max_frames
        1600 remains a documented deviation (doesn't fit 24 GB under
        Windows/WDDM at fp32).
      - Epoch-10 gate PASSED (loss_val 102.6 -> 33.6, monotone). Early curve
        tracks ~0.7 behind the vsrfe twin (33.6 vs 32.9 @ ep10) — no early
        separation; final CER decides.
      - **Result (2026-07-12): CER 28.04%** (model_avg_10, eval log
        lightning_logs/version_13 verified) vs the single-variable twin's
        26.63%. The paper's own framing is ~1.4 pt WORSE on our recording.
        **FRAMING/PREPROCESSING HYPOTHESIS FULLY EXHAUSTED**: all four
        variants sit in a 26.4–28.0 band (26.36 fixed/AV-fe, 26.63
        fixed/VSR-fe, 26.50 recrop, 28.04 whole-frame). Why whole-frame is
        slightly worse here: mouth shrinks to ~31px (vs 45–53px), lip detail
        is lost and distractor context added — furthest from the English
        frontend's training distribution, and 7.7 h of fine-tuning can't
        recover it. **Conclusion: the ~7 pt gap to 19.70% is attributed to
        the recording/speaker difference** (duration fingerprint: theirs
        8.15 s/utt vs our zundamon 6.70 s/utt — ~22% slower speech = more
        frames per mora), with max_frames 1600 (untestable at fp32 on our
        hardware) and test-split identity as minor residuals. VSR write-up
        line: **26.36%** + ablation table showing robustness to frontend
        scale, frontend modality, and 3 crop geometries.
      - Per the pre-registered rule: **AVSR arms proceed on
        rohan_preprocessed_fixed** (our best-performing video prep) + the
        19.2 kHz audio switch.
- [ ] Pre-AVSR audio switch to 19,200 Hz (paper-faithful; do AFTER the VSR
      paper-prep run settles the video prep, BEFORE the AVSR arms):
      **Implemented 2026-07-12** (all sub-items below, plus one the original
      list missed): the 1D-ResNet audio front-end hardcodes 640 samples/frame
      (conv stride 32 x avgpool 20) — at 19.2 kHz that yields 30 feat/s vs
      25 video fps and breaks fusion alignment. Fix: `aux_rate_ratio` model
      arg (default 640) threads through Encoder -> Conv1dResNet -> ResNet1D;
      at 768 only the parameter-free avgpool changes (20 -> 24), so the
      English front-end still strict-loads (unit-verified). Also
      parameterized the `T//640` truncation and the `audio_lengths/640`
      division in e2e_asr_conformer_av (ASR path untouched, default 640).
      AddNoise now resamples the noise file to the speech rate on load
      (replaces the 16 kHz assert; no duplicate noise files needed).
      AudioTransform gets sample_rate = rate_ratio x 25 from the datamodule;
      time mask scales to 0.4 s at any rate. Dataset builder:
      `preparation/build_rohan_19k2.py` (videos bit-identical from
      rohan_preprocessed_fixed; audio decimated x5 from the RAW 96 kHz wavs,
      not from the 16 kHz files). Config: `rohan_avsr_19k2.yaml`
      (root_dir + rate_ratio 768 + aux_rate_ratio 768 baked; hydra-verified).
      Unit checks passed: 640/768 output lengths, strict frontend load @768,
      full AV forward @768 finite loss with uneven batch lengths.
      1. Rebuild audio: `preprocess_rohan.py --audio-sample-rate 19200` into a
         fresh root, then run the winning video normalization from it (wavs are
         carried alongside) → the AVSR dataset (e.g.
         `rohan_preprocessed_avsr19k2`).
      2. New AVSR experiment yaml: baked `root_dir` + `data.dataset.rate_ratio: 768`.
      3. `datamodule/transforms.py`: generalize the `AddNoise` 16 kHz assert
         (line ~123) to the configured rate; resample noise files to 19.2 kHz
         (babble — used at train time via noise_prob=1.0 — and white for
         Table IV: `noise_datasets/babble_19k2.wav`, `white_19k2.wav`).
      4. Scale `AdaptiveTimeMask(6400, 16000)` (line ~170) → (7680, 19200) to
         keep the same masked duration in seconds (paper-side handling of this
         constant is unknowable — note in write-up).
      5. Re-verify hydra composition + a 1-batch smoke run before the 7 h arms.
      DONE 2026-07-12: dataset `rohan_preprocessed_avsr19k2` built + verified
      (4,600 pairs, 19,200 Hz, AV alignment same 18 ms tail as at 16 kHz);
      fast_dev_run smoke PASSED (real data, frozen gate, both frontends
      strict-loaded, finite loss). White-noise Table IV needs no new files:
      AddNoise auto-resamples `white_16k.wav` to the speech rate.
      **REVERTED for the arms 2026-07-12: 19.2 kHz @ max_frames 1000 does NOT
      fit the 24 GB RTX 4090 under Windows/WDDM at fp32** — measured ~5.8 GB
      sysmem spill (shared GPU memory counter), pace degraded to ~17 min/epoch
      (~11 h/arm), unsustainable for the deadline and a stability risk on this
      machine's RAM. The 16 kHz config is the only one with empirical proof of
      fitting (all prior AVSR runs <10 h). AVSR arms therefore run at 16 kHz
      via `rohan_avsr_fixed` (deviation #2 re-accepted, honestly documented:
      AVSR noise rows not directly comparable to the paper's 19.2 kHz Table IV;
      their ASR rows suggest 16 kHz is the MORE noise-robust setting; the
      static-vs-gated twin comparison is internal and unaffected). The 19.2 kHz
      code path (aux_rate_ratio) and dataset remain in the repo, unit-tested,
      for future hardware.
- [x] Train AVSR static → eval clean (paper reference 2.78% @ 19.2 kHz;
      ours at 16 kHz, see revert note above)
      - **Result (2026-07-12): CER 3.13%** (train version_41: frozen gate,
        800 frames, fixed 16k data, 40 eps, best val 3.84 @ ep38 — healthy;
        eval version_14: model_avg_10, clean). vs paper 2.78% @ 19.2 kHz:
        applying their own 16 kHz penalty (ASR 3.88 vs 3.34 = ~0.54 pt) the
        16 kHz-adjusted expectation is ~3.2-3.3% → **matched/beat**. Relative
        multimodal gain reproduces: ours ASR 3.87 -> AVSR 3.13 = 19% rel.
        reduction vs paper 3.34 -> 2.78 = 17% rel. **AVSR-static reproduction
        SUCCESSFUL within documented deviations.**
      - 2026-07-12 memory saga, root cause found: the AVSR model @ fp32 /
        max_frames 1000 needs ~28 GB — it spills ~5.5 GB to system RAM
        REGARDLESS of audio rate (per-process counters; my earlier 19.2k-only
        attribution was wrong), with WDDM paging degrading pace mid-run
        (12 -> 16 min/epoch even with desktop apps closed; dwm had also been
        holding 3.5 GB). June's sub-10 h runs likely ran with a lighter
        desktop + less fragmentation.
      - FINAL ARM SPEC: 16 kHz (rohan_avsr_fixed) + **max_frames 800/800**
        (fresh restart; stale 1000-frame checkpoints deleted from the exp dir
        before relaunch — the avg-10 ensemble scans that directory). New
        documented deviation: AVSR max_frames 800 vs paper's 1000 (doesn't
        fit 24 GB WDDM fp32 here); both arms identical -> twin comparison
        unaffected; expected clean-CER effect small (batch-size noise).
- [x] Train AVSR gated → eval clean
      - **Result (2026-07-12): CER 3.52% clean** vs static twin 3.13%
        (+0.39 pt clean premium). Valid run (version_42: gate trains, 800
        frames, 40 eps, best val 3.79 @ ep31 — better than static's 3.84;
        avg-10 interacts less well with a drifting gate, footnote). Gate
        VERIFIED ACTIVE: eval gate_w_audio = 0.537 (≠ 0.5), leaning audio on
        clean speech as expected. Interpretation: expected pattern — the gate
        can't help when audio is clean; the hypothesis test is the white-noise
        twin comparison @ 10/5 dB + gate_w_audio-vs-SNR mechanism curve.
- [x] Table IV white-noise sweep (ASR, AVSR-static, AVSR-gated @ clean/10/5 dB)
      — COMPLETE 2026-07-12. ASR rows (version_20/21, white noise verified):
      **26.09% @ 10 dB (paper 26.02 — near-exact), 43.42% @ 5 dB (paper
      40.13)**. Full replica table + four lessons in the write-up section:
      (1) AVSR noise-robustness claim reproduced (−59%/−64% relative vs
      paper's −54%/−60%); (2) every audio-bearing row matches → pipeline
      fidelity proven, VSR gap isolated to video data; (3) 5 dB audio still
      out-informs clean video (AVSR 15.7 < VSR 26.4); (4) gate mechanism
      works but is redundant under stationary noise at these SNRs.
      - AVSR-gated DONE (2026-07-12, white noise verified in eval logs
        version_16/17): **10.70% @ 10 dB (gate_w_audio 0.489), 16.28% @ 5 dB
        (0.443)**; with clean 3.52% (0.537) the gate weight falls
        monotonically with SNR — mechanism CONFIRMED. Numbers plausible vs
        paper (11.95/16.11 @ 19.2 kHz): 16 kHz is the more robust rate per
        their own ASR rows. No train/test noise overlap (train = babble,
        eval = white). Static @ 10/5 dB + ASR rows still pending — static
        pair is decisive for attributing robustness to the gate.
      - AVSR-static noise pair DONE (2026-07-12, version_18/19 validated:
        right ckpt, white noise, gate_w_audio = 0.5 exact): **10.79% @ 10 dB,
        15.73% @ 5 dB**. TWIN VERDICT: tie @ 10 dB (Δ0.09), static wins 5 dB
        (+0.55) and clean (+0.39) — the gate re-weights correctly (mechanism
        confirmed) but yields NO recognition benefit under stationary white
        noise. Robustness is attributable to noise-augmented AVSR training
        itself, not the gate. See memory noise-sensitive-fusion for the full
        interpretation + future-work directions. ASR white-noise rows remain
        (last pending item before write-up).
- [ ] Write up: reproduction table + static-vs-gated comparison

---

## Incident log

### 2026-07-08 — VSR run #1 invalid (150% CER): trained on the wrong dataset

**Symptom.** First `rohan_vsr_3448` run (3,448 h visual frontend) evaluated at
**150% CER** — the same pathological regime as the pre-fix 123% collapse, even
though ASR had just reproduced the paper exactly (3.87% vs 3.88%).

**Root cause.** The Phase 1 VSR command in this plan omitted the
`data.dataset.root_dir` override. `rohan_vsr_fixed.yaml` — despite its name —
did not bake in the fixed dataset path; it inherited the default from
`configs/data/dataset/rohan.yaml`, which points at the ORIGINAL broken
preprocessing `datasets/rohan_preprocessed` (un-cropped 300×300 ROI, ~30 fps).
The run therefore trained AND evaluated on out-of-distribution video. The ASR
command happened to include the override, which is why ASR was unaffected
(and ASR audio is identical across both dataset versions anyway).

**Evidence** (from the runs' own logged hparams — check these, not the command
you think you ran):

- Training log `auto_avsr/exp/lightning_logs/version_30/hparams.yaml`:
  `root_dir: C:/Users/Hiroyuki/gr_avsr/datasets/rohan_preprocessed` (wrong).
- Eval log `auto_avsr/lightning_logs/version_9/hparams.yaml`: same wrong root.
- Training curve = the known wrong-video memorization fingerprint: best
  `loss_val` 93.4 at **epoch 9**, rising afterwards — numerically matching the
  original broken-preprocessing failure (val loss 93→137, best epoch 8).
  `model_avg_10.pth` then averages epochs 50–59 (the most-overfit), producing
  gibberish longer than the references → CER > 100%.

**What this run does NOT show:** anything about the 3,448 h frontend. The
frontend-scale hypothesis (27.26% → ~19.7%) remains untested until the rerun.

**Fix (applied 2026-07-08).**
- `rohan_vsr_fixed.yaml` now hard-codes
  `data.dataset.root_dir: C:/Users/Hiroyuki/gr_avsr/datasets/rohan_preprocessed_fixed`
  (verified via hydra composition). `rohan_avsr_fixed.yaml` already had it.
- Rule going forward: **experiment configs must bake in their dataset root**;
  never depend on a CLI override for data identity. Overrides are for
  hyperparameters, not for which data a run sees.
- Stale artefacts: `exp/rohan_vsr_3448/` (~33 GB) and the version_30/version_9
  logs are garbage; delete or rename before the rerun.

**Early-detection check for the rerun:** by epoch ~10, `loss_val` must be well
below ~90 and still falling. `loss_val` ≈ 93 flattening at epoch ~9 = wrong
data; kill the run immediately instead of burning 10 h.
