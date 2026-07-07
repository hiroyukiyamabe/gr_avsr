import csv
import logging
import os

import hydra
import torch

from pytorch_lightning import Trainer
from lightning import ModelModule
from datamodule.data_module import DataModule


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    # Set modules and trainer
    if cfg.data.modality in ["audio", "video"]:
        from lightning import ModelModule
    elif cfg.data.modality == "audiovisual":
        from lightning_av import ModelModule
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    trainer = Trainer(num_nodes=1, accelerator="auto", devices=1)
    # Training and testing
    # strict=False so a pre-gate baseline checkpoint still loads: its missing
    # fusion_gate.* weights stay zero-init (neutral 0.5/0.5), which reproduces
    # the original static fusion exactly -- a faithful static-fusion baseline
    # through the same code path. Warn on any OTHER mismatch so genuine key
    # errors (e.g. a corrupt or wrong checkpoint) aren't silently masked.
    state = torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
    missing, unexpected = modelmodule.model.load_state_dict(state, strict=False)
    missing_real = [k for k in missing if not k.startswith("fusion_gate.")]
    if missing_real or unexpected:
        logging.warning("state_dict mismatch -- missing=%s unexpected=%s", missing_real, list(unexpected))
    if any(k.startswith("fusion_gate.") for k in missing):
        logging.info("fusion_gate.* absent from checkpoint -> neutral gate (static-fusion baseline)")
    results = trainer.test(model=modelmodule, datamodule=datamodule)

    # Append this run's metrics to a per-experiment CSV so an SNR sweep builds a
    # ready-to-plot table (one row per decode.snr_target) with no console
    # copy-paste. Keyed by exp_name, so gated and baseline runs stay separate.
    metrics = results[0] if results else {}
    error_key = "cer" if "cer" in metrics else "wer"
    out_csv = os.path.join(cfg.exp_dir or ".", f"snr_sweep_{cfg.exp_name}.csv")
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["snr_target", error_key, "gate_w_audio"])
        writer.writerow([
            cfg.decode.snr_target,
            metrics.get(error_key, ""),
            metrics.get("gate_w_audio", ""),
        ])
    logging.info("appended sweep row -> %s", out_csv)


if __name__ == "__main__":
    main()
