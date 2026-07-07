# Japanese AVSR Pipeline — Command Reference

Cross-lingual transfer fine-tuning (Auto-AVSR baseline, framework (c)): inherit
English front-ends, train Conformer back-ends + decoder from scratch on the
Japanese ROHAN4600 corpus (character-level Katakana, ~87 units).

All commands run from `C:\Users\Hiroyuki\gr_avsr\auto_avsr` using the project
venv Python: `C:\Users\Hiroyuki\gr_avsr\avsr\Scripts\python.exe`.

> **Hardware note (resolved 2026-06-29):** the machine previously BSOD'd under
> load due to an unstable DDR5 XMP overclock. Fixed by reverting BIOS to
> defaults + updating BIOS/firmware (see `../SYSTEM_STATUS.md`); a full
> multi-hour run completed with no crash. **`+data.num_workers=4
> +data.pin_memory=True`** is now the recommended setting; drop to
> `num_workers=0` only if instability returns. Separately, keep VRAM headroom
> (`data.max_frames` ≈ 800–1200) — at the full budget the RTX 4090 ran ~98% and
> a conv3d batch can hit a cuDNN/OOM failure.

---

## 0. One-time setup (already done)

| Step | Command | Output |
|------|---------|--------|
| Extract English front-ends | `python ../module_extractor/extractor.py` | `../module_extractor/frontends/frontend_visual_trlrs3.pth`, `frontend_audio_trlrs3.pth` |
| Preprocess ROHAN corpus | `python preparation/preprocess_rohan.py` | `datasets/rohan_preprocessed/` (3800 train / 400 val / 400 test) + label CSVs |

Extractor defaults: reads `../module_extractor/{vsr,asr}_trlrs3_base.pth`, writes
to `frontends/`. Preprocess defaults: 16 kHz audio, 25 fps, val/test = 400 each.
Add `--reencode` to re-encode video, `--limit N` to process only N utterances.

Dataset paths are wired in `configs/data/dataset/rohan.yaml`
(`root_dir: C:/Users/Hiroyuki/gr_avsr/datasets/rohan_preprocessed`), so you do
not need to pass `data.dataset.root_dir` on the CLI.

---

## 1. Fine-tune

Pick the modality. Checkpoints land in `exp/<exp_name>/`; at the end the last 10
epochs are averaged into `exp/<exp_name>/model_avg_10.pth`.

> **Run the `train.py` / `eval.py` commands directly** (in the activated venv,
> from `auto_avsr/`). Avoid wrapping them in a PowerShell script that combines
> `$ErrorActionPreference = "Stop"` with `& python ... 2>&1 | Tee-Object`: that
> turns Python's benign stderr warnings (e.g. `triton not found`) into a
> terminating `NativeCommandError` and kills the run before training starts. For
> a saved log, prefer detached `Start-Process -RedirectStandardError` (§3),
> which writes stderr to a file with no PowerShell pipeline involved.

### AVSR (audiovisual — fine-tunes BOTH front-ends) — 40 epochs
```
python train.py +experiment=rohan_avsr exp_dir=exp exp_name=rohan_avsr \
    pretrained_visual_frontend_path=../module_extractor/frontends/frontend_visual_trlrs3.pth \
    pretrained_audio_frontend_path=../module_extractor/frontends/frontend_audio_trlrs3.pth \
    +data.num_workers=0 +data.pin_memory=False
```

### VSR (video only — visual front-end) — 60 epochs
```
python train.py +experiment=rohan_vsr exp_dir=exp exp_name=rohan_vsr \
    pretrained_model_path=../module_extractor/frontends/frontend_visual_trlrs3.pth \
    +data.num_workers=0 +data.pin_memory=False
```

### ASR (audio only — audio front-end) — 60 epochs
```
python train.py +experiment=rohan_asr exp_dir=exp exp_name=rohan_asr \
    pretrained_model_path=../module_extractor/frontends/frontend_audio_trlrs3.pth \
    +data.num_workers=0 +data.pin_memory=False
```

Handy overrides (Hydra):
- `trainer.max_epochs=N` — change epoch count (note: existing key, **no** `+`).
- `+trainer.limit_train_batches=50 +trainer.limit_val_batches=20 trainer.max_epochs=1`
  — quick smoke test (a few minutes).
- `+trainer.fast_dev_run=true` — 1 train + 1 val batch, no checkpoints written.
- `optimizer.lr=...`, `data.max_frames=...` — tune LR / batch token budget.

---

## 2. Evaluate (reports CER for `lang=ja`)

Evaluate an averaged model on the 400-clip test split. **Pass
`transfer_frontend=false`**: it makes the model constructor load the full
averaged checkpoint directly (its keys match the `E2E` model). Without it, the
constructor instead tries to read a front-end-only checkpoint
(`ckpt["model_state_dict"]`) and crashes — this matters because `eval.py` reuses
`pretrained_model_path` for both construction and the final weight load.

```
# AVSR
python eval.py +experiment=rohan_avsr \
    pretrained_model_path=exp/rohan_avsr/model_avg_10.pth \
    transfer_frontend=false \
    +data.num_workers=4 +data.pin_memory=True data.max_frames_val=1200

# VSR (video only)
python eval.py +experiment=rohan_vsr \
    pretrained_model_path=exp/rohan_vsr/model_avg_10.pth \
    transfer_frontend=false \
    +data.num_workers=4 +data.pin_memory=True data.max_frames_val=1200

# ASR (audio only)
python eval.py +experiment=rohan_asr \
    pretrained_model_path=exp/rohan_asr/model_avg_10.pth \
    transfer_frontend=false \
    +data.num_workers=4 +data.pin_memory=True data.max_frames_val=1200
```

The test loop logs the metric as **`cer`** when `data.lang=ja` (character-level
edit distance) and `wer` otherwise — the same value, just the correct label.

---

## 3. Running it so it survives (long runs)

The full AVSR run is 40 epochs over 3,800 clips on 1 GPU — hours. Two options:

**Foreground in your own terminal** (see live progress, keep a log):
```
... train.py ...args... 2>&1 | Tee-Object -FilePath exp\rohan_avsr.log
```

**Detached** (survives terminal/session teardown) — PowerShell:
```powershell
$py = "C:\Users\Hiroyuki\gr_avsr\avsr\Scripts\python.exe"
Start-Process -FilePath $py -WorkingDirectory "C:\Users\Hiroyuki\gr_avsr\auto_avsr" `
  -ArgumentList @("train.py","+experiment=rohan_avsr","exp_dir=exp","exp_name=rohan_avsr",
    "pretrained_visual_frontend_path=../module_extractor/frontends/frontend_visual_trlrs3.pth",
    "pretrained_audio_frontend_path=../module_extractor/frontends/frontend_audio_trlrs3.pth",
    "+data.num_workers=0","+data.pin_memory=False") `
  -RedirectStandardOutput "exp\rohan_avsr.out.log" -RedirectStandardError "exp\rohan_avsr.err.log" `
  -WindowStyle Hidden
```
Watch progress (Lightning writes the progress bar to stderr):
`Get-Content exp\rohan_avsr.err.log -Tail 20 -Wait`

---

## Config map (where to change things)

| Want to change | File |
|---|---|
| Dataset paths, rate_ratio | `configs/data/dataset/rohan.yaml` |
| Epochs, LR, modality, max_frames per experiment | `configs/experiment/rohan_{avsr,vsr,asr}.yaml` |
| Trainer defaults (precision, grad clip, devices) | `configs/trainer/train.yaml` |
| DataLoader workers / pin_memory | `datamodule/data_module.py` (overridable via `+data.num_workers=`) |
| Checkpoint averaging (last 10 epochs) | `avg_ckpts.py` |

See `JAPANESE_AVSR.md` for the corpus layout, label CSV format, and the paper's
19,200 Hz AVSR variant (`data.dataset.rate_ratio=768`).
