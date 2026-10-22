# gr_avsr

Japanese audio-visual speech recognition via **cross-lingual transfer learning**
(Auto-AVSR baseline; Kondo & Tamura, APSIPA ASC 2025, framework (c)). English
front-ends are inherited and fine-tuned on the recorded **ROHAN4600** corpus.
On top of the reproduction, a static-vs-dynamic (gated) audio-visual fusion
twin study.

**Status (2026-10-22):** all experiments complete — reproduction (ASR/VSR/AVSR,
Table IV white-noise replica), fusion twin study, and an 8-level white-noise
SNR sweep (clean → −10 dB, 2026-10-08). Open items: seed the eval noise
(see "Gotchas" below), then the write-up.

## Documentation — read in this order

| Doc | What it covers | Current? |
|-----|----------------|----------|
| [`RESULTS.md`](RESULTS.md) | **Start here.** Final numbers: reproduction table, twin study, 8-level SNR sweep (§2.1), VSR ablations, run provenance, deviations from the paper. | ✅ current |
| [`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md) | Experimental log: exact commands for every run (PowerShell), checklist with dated results, incident log. | ✅ current |
| [`SYSTEM_STATUS.md`](SYSTEM_STATUS.md) | Software environment + code modifications for the modern stack (PL 2.6, PyAV, no fairseq) and the resolved RAM/BSOD incident. | ⚠ 2026-06-29; environment section still valid |
| [`auto_avsr/PIPELINE.md`](auto_avsr/PIPELINE.md), [`auto_avsr/JAPANESE_AVSR.md`](auto_avsr/JAPANESE_AVSR.md) | Original June pipeline: tokenizer, corpus layout, label CSV format, Hydra overrides, detached-run recipe. | ⚠ legacy — commands use the broken `rohan_preprocessed` dataset and 438 h `trlrs3` front-ends |
| [`auto_avsr/README.md`](auto_avsr/README.md) | Upstream Auto-AVSR documentation. | upstream |

## Layout

- `auto_avsr/` — training/eval code (vendored Auto-AVSR + local `espnet/`).
  - `train.py`, `eval.py` (faithful 1.0.0 port), `eval_sweep.py` (eval + appends
    a CSV row with `gate_w_audio`), `plot_snr_sweep.py`.
  - `configs/experiment/` — use the `*_fixed` configs (dataset root baked in).
  - `exp/<exp_name>/` — checkpoints; `model_avg_10.pth` is the reported model.
  - `exp/snr_sweep_*.csv`, `exp/snr_sweep_white_8lvl.png`, `exp/white_sweep_8lvl.log` — SNR sweep outputs.
- `module_extractor/` — extracts English front-ends into `frontends/`
  (`frontend_{visual,audio}_3448.pth` are the current ones).
- `datasets/rohan_preprocessed_fixed/` — **canonical** dataset (25 fps, 96×96
  mouth ROI, 16 kHz, 3800/400/400). Other `rohan_preprocessed*` dirs are
  ablations or the broken original.
- `noise_datasets/white_16k.wav` — NoiseX-92 white noise for eval (training uses babble).
- `original_checkpoints/` — downloaded Auto-AVSR model-zoo checkpoints.
- `avsr/` — Python 3.12 venv (`avsr\Scripts\python.exe`).
- `auto_avsr_original_do_not_touch/` — pristine upstream copies; read-only.

## Gotchas (read before running anything)

1. **Shell is Windows PowerShell 5.1** — see the shell notes in `REPRODUCTION_PLAN.md` § Commands.
2. **Default dataset root is the broken one.** `configs/data/dataset/rohan.yaml`
   points at `datasets/rohan_preprocessed` (un-cropped 300 px, ~30 fps → VSR
   CER > 100 %). Always use an experiment config that bakes in its root
   (`rohan_avsr_fixed`, `rohan_vsr_fixed`, …) or pass
   `data.dataset.root_dir=../datasets/rohan_preprocessed_fixed`. Verify via the
   run's logged `hparams.yaml`, not the command you think you ran.
3. **Logged `cer` is a fraction** (0.1048 = 10.48 %), in both Lightning logs and sweep CSVs.
4. **Two `lightning_logs` dirs with overlapping version numbers:**
   training runs log to `auto_avsr/exp/lightning_logs/version_N`, evals to
   `auto_avsr/lightning_logs/version_N`. "version_30" exists in both and means
   different runs — always say which dir.
5. **Eval noise is unseeded.** `AddNoise.forward` (`auto_avsr/datamodule/transforms.py`)
   picks a random offset into the noise file per utterance, so noisy CERs vary
   by ~0.3 pt between re-evals of the same checkpoint (clean is exact).
   Static-vs-gated gaps ≲ 0.5 pt are not interpretable.
6. **`eval_sweep.py` appends** to `exp/snr_sweep_<exp_name>.csv`. Move/rename the
   existing CSV before a new sweep, or rows from different sweeps get mixed.
7. VRAM: AVSR at fp32 needs `max_frames` ≤ 800 on the 24 GB RTX 4090 under
   Windows/WDDM (1000 spills to system RAM). `num_workers=2` is the tested setting.
