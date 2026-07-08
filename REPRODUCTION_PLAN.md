# Plan: Clean Tamura Reproduction (3,448 h frontends) + Dynamic-Fusion Swap

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

1. Own ROHAN4600 recordings (theirs are unpublished); 3,800 train sentences vs
   their 3,400.
2. Audio 16 kHz + `rate_ratio=640` (exact A/V sync at 25 fps) vs their 19,200 Hz
   + ratio 768 — their workaround for a padding problem we don't have; their own
   Table III shows ~0.5 pt effect on ASR.
3. Central 200→96 mouth crop vs their landmark-tracked 96×96 ROI. **Main suspect
   if VSR still misses 19.70%.**
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
- [ ] Train VSR → eval (target ≈ 19.7%; if ≥ ~24%, run the modality-matched
      `vsr_trlrwlrs2lrs3vox2avsp` frontend ablation before touching the crop)
      - **First attempt (2026-07-08) INVALID — 150% CER, not a frontend result.**
        The plan's original VSR command omitted `data.dataset.root_dir`, so the
        run trained AND evaluated on the broken `rohan_preprocessed` videos
        (old 300px/30fps). Training curve = the known memorization collapse
        (best loss_val 93.4 @ epoch 9, then rising — same as the old 123% run).
        Fix: fixed root_dir is now BAKED INTO `rohan_vsr_fixed.yaml`; commands
        above are correct as written. `exp/rohan_vsr_3448/` from the bad run
        (~33 GB of checkpoints) can be deleted before rerunning.
- [ ] Train AVSR static → eval clean (target ≈ 2.8%)
- [ ] Train AVSR gated → eval clean
- [ ] Table IV white-noise sweep (ASR, AVSR-static, AVSR-gated @ clean/10/5 dB)
- [ ] Write up: reproduction table + static-vs-gated comparison
