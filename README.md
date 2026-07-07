# gr_avsr

Japanese audio-visual speech recognition via **cross-lingual transfer learning**
(Auto-AVSR baseline; Kondo & Tamura, APSIPA ASC 2025, framework (c)). English
front-ends are inherited and fine-tuned on the recorded **ROHAN4600** corpus.

## Documentation

| Doc | What it covers |
|-----|----------------|
| [`SYSTEM_STATUS.md`](SYSTEM_STATUS.md) | **Start here.** Software environment, code modifications, and the open hardware-stability issue. |
| [`auto_avsr/PIPELINE.md`](auto_avsr/PIPELINE.md) | Copy-paste command reference: front-end extraction, preprocessing, train (AVSR/VSR/ASR), eval. |
| [`auto_avsr/JAPANESE_AVSR.md`](auto_avsr/JAPANESE_AVSR.md) | Method, model architecture, corpus layout, label CSV format. |
| [`auto_avsr/README.md`](auto_avsr/README.md) | Upstream Auto-AVSR documentation. |

## Layout

- `auto_avsr/` — training/eval code (vendored Auto-AVSR + local `espnet/`).
- `module_extractor/` — extracts English front-ends into `frontends/`.
- `datasets/rohan_preprocessed/` — preprocessed ROHAN data (3800/400/400).
- `avsr/` — Python 3.12 venv (`avsr\Scripts\python.exe`).

## ⚠ Before running

This machine currently BSOD-crashes under heavy training load — traced to
**unstable system memory (DDR5 XMP)**, not the code. Until it is fixed, append
**`+data.num_workers=0 +data.pin_memory=False`** to every `train.py` / `eval.py`
command. See [`SYSTEM_STATUS.md`](SYSTEM_STATUS.md) for the diagnosis and fix.

## Quick start (AVSR fine-tune)

From `auto_avsr/`, using `..\avsr\Scripts\python.exe`:

```
python train.py +experiment=rohan_avsr exp_dir=exp exp_name=rohan_avsr \
    pretrained_visual_frontend_path=../module_extractor/frontends/frontend_visual_trlrs3.pth \
    pretrained_audio_frontend_path=../module_extractor/frontends/frontend_audio_trlrs3.pth \
    +data.num_workers=0 +data.pin_memory=False
```

See [`auto_avsr/PIPELINE.md`](auto_avsr/PIPELINE.md) for VSR/ASR, evaluation, and
long-run (detached) recipes.
