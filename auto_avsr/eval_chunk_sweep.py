"""Chunk-wise corruption sweep for the gate diagnostic (Phase 2c, strong form, Part B).

Inference only (CLAUDE.md section 5.2). Seeded chunks of the audio and/or video
stream of every test utterance are corrupted (decode.chunk_condition, settings
in decode.chunk); the per-frame masks are logged with the per-frame audio gate
weight. Same model, checkpoint, decoding and aggregate metrics as eval.py: the
unchanged test_step of lightning_av.py (AVSR) or lightning.py (ASR, VSR) is
wrapped, per-utterance values are increments of its running totals, and the gate
is read with a forward hook on ``fusion_gate`` (observes, never alters).

Outputs (one new directory per run; refuses to overwrite):
  <chunk_out_dir>/<exp_name>/<condition>/rows.csv    one row per utterance
  <chunk_out_dir>/<exp_name>/<condition>/frames.npz  per-frame gate weight (AVSR)
                                                     and both masks, all utterances
  <chunk_out_dir>/<exp_name>/<condition>/meta.json   provenance, aggregates, checks
chunk_out_dir defaults to <exp_dir>/chunk_sweep.

Real runs abort unless the git tree is clean (untracked eval logs under
auto_avsr/lightning_logs/ excepted: earlier runs of the same sweep write them).
Smoke tests pass +chunk_smoke=true and +chunk_out_dir=<dir outside the repo>.
"""
import csv
import datetime
import hashlib
import json
import logging
import os
import subprocess

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from datamodule.av_dataset import utterance_id
from datamodule.chunk_corruption import CONDITIONS, resolve, sha256_text_file
from datamodule.data_module import DataModule


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Masks the row statistics are split by: the corrupted streams, except that the
# placebo is read with both drawn masks as if both had been applied.
READ_AS = {"audio_chunks": (True, False), "video_chunks": (False, True),
           "both_chunks": (True, True), "placebo": (True, True)}
CATEGORIES = ("audio_only", "video_only", "both", "neither")
CSV_COLUMNS = (
    ["arm", "condition", "utt_id", "n_frames", "frac_audio_mask", "frac_video_mask"]
    + [f"n_{c}" for c in CATEGORIES]
    + [f"gate_w_audio_{c}" for c in CATEGORIES]
    + ["gate_w_audio_mean", "edit_distance", "ref_len", "audio_sha1", "video_sha1", "mask_sha1"]
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _sha1(tensor):
    return hashlib.sha1(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def _git_state():
    """HEAD and dirtiness of the whole repo, ignoring untracked eval logs."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                capture_output=True, text=True, cwd=REPO_ROOT).stdout.splitlines()
        dirty = [s for s in status if not (s.startswith("?? ") and s[3:].strip('"').startswith("auto_avsr/lightning_logs/"))]
        return {"head": head, "dirty": bool(dirty), "dirty_paths": dirty}
    except OSError:
        return {"head": None, "dirty": None, "dirty_paths": None}


def categories(audio_mask, video_mask, condition):
    a_on, v_on = READ_AS[condition]
    a = audio_mask & a_on
    v = video_mask & v_on
    return {"audio_only": a & ~v, "video_only": v & ~a, "both": a & v, "neither": ~a & ~v}


def chunk_module(base):
    class ChunkSweepModule(base):
        """Records per-utterance stats around the unchanged parent ``test_step``."""

        def __init__(self, cfg, utt_list, condition):
            super().__init__(cfg)
            self.utt_list = utt_list
            self.condition = condition
            self.modality = cfg.data.modality
            self.check_neutral_gate = (
                self.modality == "audiovisual" and bool(cfg.model.audiovisual_backbone.fusion_gate_freeze)
            )
            self.rows, self.frames = [], []
            self.max_raw_offset = 0
            self._gate = None
            if self.modality == "audiovisual":
                self.model.fusion_gate.register_forward_hook(self._capture_gate)

        def _capture_gate(self, module, inputs, output):
            self._gate = output.detach()

        def test_step(self, sample, sample_idx):
            # batch_size=None + sequential sampler: sample_idx is the dataset index.
            _, rel_path, n_label, token_id = self.utt_list[sample_idx]
            utt = utterance_id(rel_path)
            if not torch.equal(sample["target"].cpu(), token_id) or sample["utt_id"] != utt:
                raise RuntimeError(f"sample {sample_idx} does not match label row {rel_path}")
            n = int(sample["n_frames"])
            audio_mask = sample["audio_mask"].cpu().numpy()
            video_mask = sample["video_mask"].cpu().numpy()
            video_type = sample["video_type"].cpu().numpy()
            audio = sample["audio"] if self.modality == "audiovisual" else sample["input"] if self.modality == "audio" else None
            video = sample["video"] if self.modality == "audiovisual" else sample["input"] if self.modality == "video" else None

            # Frame alignment (CLAUDE.md 5.2): 1 frame = 40 ms = 640 samples = 1 video frame.
            problems = []
            if n != n_label or len(audio_mask) != n or len(video_mask) != n:
                problems.append(f"label {n_label}, item {n}, masks {len(audio_mask)}/{len(video_mask)}")
            if video is not None and video.shape[0] != n:
                problems.append(f"{video.shape[0]} video frames")
            if audio is not None:
                raw_offset = int(sample["audio_raw_len"]) - 640 * n
                self.max_raw_offset = max(self.max_raw_offset, abs(raw_offset))
                if abs(raw_offset) >= 640:
                    problems.append(f"raw audio {sample['audio_raw_len']} samples vs 640 x {n}")
                if self.modality == "audiovisual" and audio.shape[0] != 640 * n:
                    problems.append(f"model audio input {audio.shape[0]} samples vs 640 x {n}")
            if problems:
                raise RuntimeError(f"{utt}: frame misalignment: {'; '.join(problems)}")

            edits0, len0 = self.total_edit_distance, self.total_length
            self._gate = None
            super().test_step(sample, sample_idx)

            row = {
                "arm": self.cfg.exp_name, "condition": self.condition, "utt_id": utt, "n_frames": n,
                "frac_audio_mask": float(audio_mask.mean()), "frac_video_mask": float(video_mask.mean()),
                "edit_distance": int(self.total_edit_distance - edits0),
                "ref_len": int(self.total_length - len0),
                "audio_sha1": _sha1(audio) if audio is not None else "",
                "video_sha1": _sha1(video) if video is not None else "",
                "mask_sha1": hashlib.sha1(audio_mask.tobytes() + video_mask.tobytes() + video_type.tobytes()).hexdigest(),
            }
            cats = categories(audio_mask, video_mask, self.condition)
            for c in CATEGORIES:
                row[f"n_{c}"] = int(cats[c].sum())
            w_audio = None
            if self.modality == "audiovisual":
                w_audio = self._gate[0, :, 1].float().cpu().numpy()
                if w_audio.shape[0] != n:
                    raise RuntimeError(f"{utt}: frame misalignment: gate trace has {w_audio.shape[0]} frames, expected {n}")
                if self.check_neutral_gate and not np.all(w_audio == 0.5):
                    raise RuntimeError(
                        f"frozen-gate arm reads != 0.5 on {utt} (min {w_audio.min()!r}, max {w_audio.max()!r}): "
                        "check the checkpoint and loading"
                    )
                # Same reduction the parent adds to total_audio_weight.
                row["gate_w_audio_mean"] = self._gate[..., 1].sum().item() / n
                for c in CATEGORIES:
                    row[f"gate_w_audio_{c}"] = float(w_audio[cats[c]].astype(np.float64).mean()) if cats[c].any() else ""
            else:
                row["gate_w_audio_mean"] = ""
                for c in CATEGORIES:
                    row[f"gate_w_audio_{c}"] = ""
            self.rows.append(row)
            self.frames.append((utt, n, w_audio, audio_mask, video_mask, video_type))

    return ChunkSweepModule


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    condition = cfg.decode.get("chunk_condition")
    if condition not in CONDITIONS:
        raise ValueError(f"set decode.chunk_condition to one of {sorted(CONDITIONS)}")
    if cfg.decode.get("noise_seed") is None:
        raise ValueError("set decode.noise_seed: arms must see identical, reproducible corruption")
    if cfg.decode.snr_target != 999999:
        raise ValueError("decode.snr_target must stay 999999 (clean outside the chunks)")
    modality = cfg.data.modality
    if modality == "audiovisual":
        from lightning_av import ModelModule
    else:
        from lightning import ModelModule

    smoke = bool(cfg.get("chunk_smoke") or False)
    out_root = os.path.abspath(cfg.get("chunk_out_dir") or os.path.join(cfg.exp_dir or ".", "chunk_sweep"))
    out_dir = os.path.join(out_root, cfg.exp_name, condition)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}; move it first")
    git = _git_state()
    if smoke:
        if os.path.commonpath([out_root, REPO_ROOT]) == REPO_ROOT:
            raise ValueError("smoke runs must write outside the repo (+chunk_out_dir=<scratch dir>)")
    else:
        if git["dirty"] is not False:
            raise RuntimeError(f"git tree is not clean (pre-registration, CLAUDE.md 5.2): {git['dirty_paths']}")
        if cfg.get("limit_test_batches"):
            raise ValueError("real runs use the full test set; limit_test_batches is for smoke tests")

    datamodule = DataModule(cfg)
    dataloader = datamodule.test_dataloader()
    utt_list = dataloader.dataset.list
    modelmodule = chunk_module(ModelModule)(cfg, utt_list, condition)
    # Strict, as in eval.py.
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
    if (edits, ref_chars) != (modelmodule.total_edit_distance, modelmodule.total_length):
        raise RuntimeError("per-utterance rows do not sum to the aggregate totals")
    if modality == "audiovisual" and sum(r["n_frames"] for r in rows) != modelmodule.total_frames:
        raise RuntimeError("per-utterance frames do not sum to the aggregate frame count")

    os.makedirs(out_dir, exist_ok=False)
    with open(os.path.join(out_dir, "rows.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    utts, ns, ws, ams, vms, vts = zip(*modelmodule.frames)
    arrays = {
        "utt_ids": np.array(utts), "n_frames": np.array(ns, dtype=np.int32),
        "offsets": np.concatenate([[0], np.cumsum(ns)]).astype(np.int64),
        "audio_mask": np.concatenate(ams), "video_mask": np.concatenate(vms), "video_type": np.concatenate(vts),
    }
    if modality == "audiovisual":
        arrays["gate_w_audio"] = np.concatenate(ws).astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, "frames.npz"), **arrays)

    chunk_cfg = cfg.decode.chunk
    meta = {
        "arm": cfg.exp_name,
        "condition": condition,
        "modality": modality,
        "corrupted": dict(zip(("audio", "video"), CONDITIONS[condition])),
        "smoke": smoke,
        "snr_target": cfg.decode.snr_target,
        "noise_seed": cfg.decode.noise_seed,
        "chunk": OmegaConf.to_container(chunk_cfg, resolve=True),
        "chunk_noise_sha256": _sha256(resolve(chunk_cfg.audio.noise_filename)),
        "occluder_split_sha256": sha256_text_file(resolve(chunk_cfg.video.occluder_split)),
        "pretrained_model_path": os.path.abspath(cfg.pretrained_model_path),
        "checkpoint_sha256": _sha256(cfg.pretrained_model_path),
        "fusion_gate_freeze": (bool(cfg.model.audiovisual_backbone.fusion_gate_freeze)
                               if modality == "audiovisual" else None),
        "dataset_root": cfg.data.dataset.root_dir,
        "test_file": cfg.data.dataset.test_file,
        "n_utterances": len(rows),
        "limit_test_batches": cfg.get("limit_test_batches"),
        "cer_logged": metrics.get("cer"),
        "gate_w_audio_logged": metrics.get("gate_w_audio"),
        "cer_from_rows": edits / ref_chars if ref_chars else None,
        "edit_distance_total": edits,
        "ref_chars_total": ref_chars,
        "frames_total": int(sum(ns)),
        "max_abs_raw_audio_offset_samples": modelmodule.max_raw_offset if modality != "video" else None,
        "eval_log_dir": trainer.log_dir,
        "git": git,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
