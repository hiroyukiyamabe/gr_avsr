"""Per-utterance SNR sweep for the gate diagnostic (Phase 2c, weak form).

Inference only. Same model, checkpoint, decoding and aggregate metrics as
eval_sweep.py; additionally writes one row per test utterance (gate weight,
character edits, reference length) and the per-frame audio gate trace for the
fixed utterances in decode.trace_utterances. See CLAUDE.md section 5.1.

Outputs (one new set per run; refuses to overwrite):
  <per_utt_out_dir>/<exp_name>/snr_<snr>.csv          one row per utterance
  <per_utt_out_dir>/<exp_name>/snr_<snr>_traces.json  per-frame audio weights
  <per_utt_out_dir>/<exp_name>/snr_<snr>_meta.json    provenance + aggregates
per_utt_out_dir defaults to <exp_dir>/per_utt_sweep.

Requires decode.noise_seed so that every arm sees the identical noisy input.
"""
import csv
import datetime
import hashlib
import json
import logging
import os
import subprocess

import hydra
import torch

from pytorch_lightning import Trainer
from datamodule.av_dataset import utterance_id
from datamodule.data_module import DataModule
from lightning_av import ModelModule


CSV_COLUMNS = [
    "arm", "snr_target", "utt_id", "n_frames",
    "gate_w_audio_mean", "gate_w_audio_std", "gate_w_audio_min", "gate_w_audio_max",
    "edit_distance", "ref_len", "audio_sha1",
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _git_state():
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=root).stdout.strip())
        return {"head": head, "dirty": dirty}
    except OSError:
        return {"head": None, "dirty": None}


class PerUtteranceModule(ModelModule):
    """Records per-utterance stats around the unchanged parent ``test_step``.

    Edits, reference length and frame count are the increments of the parent's
    running totals, so the aggregate CER and gate_w_audio come from exactly the
    same code as eval_sweep.py. The gate weights are read with a forward hook on
    ``fusion_gate``, which observes the output without altering it.
    """

    def __init__(self, cfg, utt_list, trace_ids):
        super().__init__(cfg)
        self.utt_list = utt_list
        self.trace_ids = set(trace_ids)
        # Static arm (frozen zero-init gate) must read exactly 0.5 everywhere.
        self.check_neutral_gate = bool(cfg.model.audiovisual_backbone.fusion_gate_freeze)
        self.rows = []
        self.traces = {}
        self._gate = None
        self.model.fusion_gate.register_forward_hook(self._capture_gate)

    def _capture_gate(self, module, inputs, output):
        self._gate = output.detach()

    def test_step(self, sample, sample_idx):
        # batch_size=None + sequential sampler: sample_idx is the dataset index.
        # Verify against the label row so a misalignment cannot go unnoticed.
        _, rel_path, _, token_id = self.utt_list[sample_idx]
        if not torch.equal(sample["target"].cpu(), token_id):
            raise RuntimeError(f"sample {sample_idx} does not match label row {rel_path}")
        utt = utterance_id(rel_path)
        audio_sha1 = hashlib.sha1(sample["audio"].cpu().contiguous().numpy().tobytes()).hexdigest()

        edits0, len0, frames0 = self.total_edit_distance, self.total_length, self.total_frames
        self._gate = None
        super().test_step(sample, sample_idx)

        n_frames = self.total_frames - frames0
        w_audio = self._gate[0, :, 1]
        if w_audio.shape[0] != n_frames:
            raise RuntimeError(f"{utt}: gate trace has {w_audio.shape[0]} frames, expected {n_frames}")
        if self.check_neutral_gate and not torch.all(w_audio == 0.5):
            raise RuntimeError(
                f"frozen-gate arm reads != 0.5 on {utt} (min {w_audio.min().item()!r}, "
                f"max {w_audio.max().item()!r}): check the checkpoint and loading"
            )
        self.rows.append({
            "arm": self.cfg.exp_name,
            "snr_target": self.cfg.decode.snr_target,
            "utt_id": utt,
            "n_frames": n_frames,
            # Same reduction the parent adds to total_audio_weight.
            "gate_w_audio_mean": self._gate[..., 1].sum().item() / n_frames,
            "gate_w_audio_std": torch.std(w_audio, correction=0).item(),
            "gate_w_audio_min": w_audio.min().item(),
            "gate_w_audio_max": w_audio.max().item(),
            "edit_distance": int(self.total_edit_distance - edits0),
            "ref_len": int(self.total_length - len0),
            "audio_sha1": audio_sha1,
        })
        if utt in self.trace_ids:
            self.traces[utt] = w_audio.cpu().tolist()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    if cfg.data.modality != "audiovisual":
        raise ValueError("eval_sweep_utt.py is for the audiovisual arms only")
    if cfg.decode.get("noise_seed") is None:
        raise ValueError("set decode.noise_seed: arms must see identical, reproducible noise")

    snr = cfg.decode.snr_target
    out_dir = os.path.join(
        cfg.get("per_utt_out_dir") or os.path.join(cfg.exp_dir or ".", "per_utt_sweep"),
        cfg.exp_name,
    )
    stem = os.path.join(out_dir, f"snr_{snr}")
    out_csv, out_traces, out_meta = stem + ".csv", stem + "_traces.json", stem + "_meta.json"
    existing = [p for p in (out_csv, out_traces, out_meta) if os.path.exists(p)]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing}; move them first")
    os.makedirs(out_dir, exist_ok=True)

    datamodule = DataModule(cfg)
    dataloader = datamodule.test_dataloader()
    utt_list = dataloader.dataset.list
    test_ids = {utterance_id(row[1]) for row in utt_list}
    trace_ids = list(cfg.decode.get("trace_utterances") or [])
    missing = [u for u in trace_ids if u not in test_ids]
    if missing:
        raise ValueError(f"trace utterances not in the test set: {missing}")

    modelmodule = PerUtteranceModule(cfg, utt_list, trace_ids)
    # Strict, as in eval.py: both twin arms were trained through the gated code
    # path, so a missing fusion_gate.* key means the wrong checkpoint.
    modelmodule.model.load_state_dict(
        torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
    )
    trainer = Trainer(
        num_nodes=1, accelerator="auto", devices=1,
        limit_test_batches=cfg.get("limit_test_batches") or 1.0,
    )
    results = trainer.test(model=modelmodule, dataloaders=dataloader)
    metrics = results[0] if results else {}

    rows = modelmodule.rows
    edits = sum(r["edit_distance"] for r in rows)
    ref_chars = sum(r["ref_len"] for r in rows)
    frames = sum(r["n_frames"] for r in rows)
    # Per-utterance rows must add up to the totals the aggregate CER is built from.
    if (edits, ref_chars, frames) != (
        modelmodule.total_edit_distance, modelmodule.total_length, modelmodule.total_frames
    ):
        raise RuntimeError("per-utterance rows do not sum to the aggregate totals")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_traces, "w", encoding="utf-8") as f:
        json.dump({"arm": cfg.exp_name, "snr_target": snr, "gate_w_audio": modelmodule.traces}, f)

    noise_path = cfg.decode.get("noise_filename")
    meta = {
        "arm": cfg.exp_name,
        "snr_target": snr,
        "noise_filename": os.path.abspath(noise_path) if noise_path else None,
        "noise_sha256": _sha256(noise_path) if noise_path else None,
        "noise_seed": cfg.decode.noise_seed,
        "pretrained_model_path": os.path.abspath(cfg.pretrained_model_path),
        "checkpoint_sha256": _sha256(cfg.pretrained_model_path),
        "fusion_gate_freeze": bool(cfg.model.audiovisual_backbone.fusion_gate_freeze),
        "dataset_root": cfg.data.dataset.root_dir,
        "test_file": cfg.data.dataset.test_file,
        "n_utterances": len(rows),
        "limit_test_batches": cfg.get("limit_test_batches"),
        "cer_logged": metrics.get("cer"),
        "gate_w_audio_logged": metrics.get("gate_w_audio"),
        "cer_from_rows": edits / ref_chars if ref_chars else None,
        "edit_distance_total": edits,
        "ref_chars_total": ref_chars,
        "frames_total": frames,
        "trace_utterances": trace_ids,
        "eval_log_dir": trainer.log_dir,
        "git": _git_state(),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("wrote %s (+ traces, meta)", out_csv)


if __name__ == "__main__":
    main()
