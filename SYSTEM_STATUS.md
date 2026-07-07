# System Status — gr_avsr

_Last updated: 2026-06-29_

Brief documentation of the current state of the Japanese AVSR project: the
software environment, the modifications made to run on the modern stack, and a
hardware-stability issue diagnosed on 2026-06-27 (resolved).

> **✅ Milestone (2026-06-29):** the AVSR cross-lingual transfer fine-tune is
> **complete** — 40 epochs, averaged into `auto_avsr/exp/rohan_avsr/model_avg_10.pth`,
> evaluating at **4.2% CER** on the 400-clip ROHAN4600 test split. The metric is
> character-level because `data.lang=ja` (Japanese ASR convention); it is now
> logged as `cer` (was mislabeled `wer`).

---

## 1. Project

Japanese audio-visual speech recognition via **cross-lingual transfer learning**
(Auto-AVSR baseline; Kondo & Tamura, APSIPA ASC 2025, framework (c)). English
front-ends are inherited and the Conformer back-ends + decoder are fine-tuned on
the recorded **ROHAN4600** corpus (character-level Katakana, ~87 units).

- Code: `auto_avsr/` (vendored Auto-AVSR + a local `espnet/` package — not pip).
- Front-end extractor: `module_extractor/`.
- Preprocessed data: `datasets/rohan_preprocessed/` (3800 train / 400 val / 400 test).
- **Command reference: [`auto_avsr/PIPELINE.md`](auto_avsr/PIPELINE.md).**

---

## 2. Software environment

| Component | Version / notes |
|---|---|
| OS | Windows 11 Pro (26200) |
| Python | 3.12 |
| venv | `C:\Users\Hiroyuki\gr_avsr\avsr` (Python at `avsr\Scripts\python.exe`) |
| PyTorch | 2.12.1 + cu126 |
| torchvision | 0.27.1 + cu126 (legacy `io.read_video`/`write_video` removed) |
| torchaudio | 2.11.0 (routes load/save through torchcodec — not installed) |
| pytorch-lightning | 2.6.5 |
| GPU | 1× CUDA GPU |
| Media I/O | **PyAV** (video), **soundfile** (audio) — replace the removed torch media APIs |
| Other deps | hydra-core, omegaconf, six, editdistance |

---

## 3. Code modifications (to run on the modern stack)

These were required because the vendored code targeted an older stack. All are
in place and valid.

- **Media APIs:** video decode/encode via PyAV, audio I/O via soundfile, across
  `preparation/preprocess_rohan.py`, `datamodule/av_dataset.py`,
  `datamodule/transforms.py`, `preparation/utils.py`. Resampling stays on
  `torchaudio.functional.resample` (pure PyTorch).
- **Lightning 2.x:** `DDPPlugin`→`DDPStrategy` (DDP only when >1 GPU),
  `ckpt_path` moved to `trainer.fit`, `gpus`→`accelerator/devices`,
  `replace_sampler_ddp`→`use_distributed_sampler`; epoch-sampler hook updated.
- **fairseq removed:** `datamodule/samplers.py` uses a local `batch_by_size`
  (fairseq does not build on this Windows/py3.12/torch2.12 env — do not install).
- **Windows spawn pickling:** `datamodule/transforms.py` uses module-level
  functions instead of lambdas so DataLoader workers can pickle the pipeline.
- **PyAV in workers:** `av_dataset.load_video` sets `stream.thread_type="NONE"`
  (default multithreaded decode segfaults inside Windows spawn workers).
- **DataLoader config:** `data_module.py` reads `num_workers`/`pin_memory` from
  `cfg.data`; `persistent_workers` auto-on when workers > 0.
- **PyTorch 2.6 load:** `avg_ckpts.py` uses `torch.load(..., weights_only=False)`
  (checkpoints embed an OmegaConf DictConfig).
- **PyTorch 2.6 resume:** `train.py` calls `trainer.fit(..., weights_only=False)`.
  Resuming from a checkpoint failed under the torch 2.6 `weights_only=True`
  default (checkpoints embed an OmegaConf DictConfig + numpy scalars in the
  callback/metric state). This path was first exercised during a mid-run resume.
- **CER labelling:** `lightning_av.py` / `lightning.py` log the test metric as
  `cer` when `data.lang=ja` (character-level edit distance) and `wer` otherwise.
  Value is unchanged — only the label was corrected.

Verified: preprocessing completed; all imports (incl. espnet) load; the full
40-epoch AVSR fine-tune ran to completion (with a mid-run resume) and evaluated
at 4.2% CER.

---

## 4. Hardware stability issue — diagnosed 2026-06-27, RESOLVED 2026-06-29

> **✅ Resolved:** after reverting BIOS to defaults (XMP/EXPO off) and updating
> the BIOS + firmware, the machine ran the full multi-hour AVSR fine-tune **with
> no BSOD** — the memory-corruption bugchecks did not recur. The fix held. The
> `num_workers=0` safe mode is no longer required; `num_workers=4` runs cleanly.
> (One unrelated GPU-side interruption did occur — see §5.)

The machine **BSOD-crashed under heavy training load.** Investigation traced
this to **unstable system memory, not the software**:

- **5 BSODs in 2 days, 4 different bugcheck codes** — `0xA`
  IRQL_NOT_LESS_OR_EQUAL, `0x50` PAGE_FAULT_IN_NONPAGED_AREA (×2), `0x1E`
  KMODE_EXCEPTION_NOT_HANDLED, `0x3B` SYSTEM_SERVICE_EXCEPTION (last two with
  `0xC0000005` access violations). A *variety* of memory-corruption bugchecks is
  the signature of unreliable RAM, not a single faulty driver.
- **WHEA-Logger Id 17** corrected PCIe AER errors on Intel root ports (ASUS board).
- Crashed even running **standalone in the user's own terminal** with
  **`num_workers=0`** → rules out Claude Code, the DataLoader workers, and the
  AVSR code itself. The code is exonerated.
- **RAM:** 2× Corsair `CMK64GX5M2B5600Z40` (DDR5, 32 GB each, 64 GB total)
  running at **5600 MHz = its XMP/EXPO rating** (JEDEC default is 4800). An
  unstable memory overclock is the prime suspect.
- Both NVMe SSDs (Samsung 990 PRO 2TB, BIWIN X570 1TB) report **Healthy**.

### Remediation (in order)
1. **Disable XMP/EXPO in BIOS** (run RAM at JEDEC default) → retest. Free, fast,
   resolves a large fraction of exactly-this-pattern cases.
2. If still crashing → **MemTest86** (bootable USB, several passes); then test
   each DIMM individually / in different slots.
3. **Update motherboard BIOS + Intel chipset drivers** (improves DDR5 training
   stability; often clears the PCIe AER warnings too).
4. To re-enable XMP later: do so after a BIOS update; if still marginal, a small
   SoC/VCCSA voltage bump or backing the kit to ~5200 MHz usually stabilizes it.

### Resolution (2026-06-29)
BIOS reverted to defaults (XMP/EXPO off) + BIOS/firmware updated. The full
40-epoch AVSR run completed with **no system crash**, confirming the fix.
`+data.num_workers=4 +data.pin_memory=True` is now the recommended setting
(faster than the old `num_workers=0` safe mode).

---

## 5. GPU-side interruption (distinct from the RAM issue) — 2026-06-29

The first attempt at the full run **crashed at epoch 29/40**, but this was a
**GPU-memory event, not a system crash** (the OS stayed up; no BSOD):

- Primary error: `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED` in
  the video front-end `conv3d` (`conv3d_extractor.py`). The follow-on
  `cudaErrorUnknown` during Lightning teardown was secondary fallout from the
  poisoned CUDA context.
- Cause: **VRAM at ~98%** (24.1 / 24.6 GB on the RTX 4090). Batches are bucketed
  by a token budget (`max_frames`), so an unlucky batch of long clips pushed the
  cuDNN conv3d workspace past the limit — intermittent, which is why it survived
  28 epochs first.

**Mitigation (worked):** resumed from `last-v2.ckpt` with
`data.max_frames=800 data.max_frames_val=800` for headroom; epochs 30–40 then
completed cleanly. Note: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is
**not supported on Windows** (silently ignored) — rely on `max_frames` instead.

**Resume note:** resuming required the `train.py` `weights_only=False` fix (§3);
the averaged checkpoint is a pure-tensor `state_dict`, so `eval.py` loads it
fine. For AVSR eval, still pass the front-end paths (`transfer_frontend=true`
makes the constructor load them before `eval.py` overwrites all weights).
