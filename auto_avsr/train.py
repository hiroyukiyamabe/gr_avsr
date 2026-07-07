import os
import hydra
import logging

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from avg_ckpts import ensemble
from datamodule.data_module import DataModule


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()

    dirpath = os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None
    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max",
        dirpath=dirpath,
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    # Also keep the single best-validation checkpoint. The last-10 average
    # (model_avg_10.pth) captures the most-overfit epochs, which is fatal for
    # low-data VSR; "best-*.ckpt" lets us evaluate the best-generalising epoch.
    best_val = ModelCheckpoint(
        monitor="loss_val",
        mode="min",
        dirpath=dirpath,
        save_last=False,
        filename="best-{epoch}-{loss_val:.3f}",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, best_val, lr_monitor]

    # Set modules and trainer
    if cfg.data.modality in ["audio", "video"]:
        from lightning import ModelModule
    elif cfg.data.modality == "audiovisual":
        from lightning_av import ModelModule
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    # Lightning 2.x: DDPPlugin -> DDPStrategy. Only use DDP with >1 GPU; fall
    # back to "auto" for single-GPU / CPU so a process group isn't forced.
    strategy = (
        DDPStrategy(find_unused_parameters=False)
        if cfg.gpus and cfg.gpus > 1
        else "auto"
    )
    trainer = Trainer(
        **cfg.trainer,
        #logger=WandbLogger(name=cfg.exp_name, project="auto_avsr"),
        callbacks=callbacks,
        strategy=strategy,
    )

    # weights_only=False: PyTorch 2.6 defaults torch.load to weights_only=True,
    # which rejects our own checkpoints on resume (they embed an OmegaConf
    # DictConfig and numpy scalars in the callback/metric state). The checkpoint
    # is trusted (we wrote it), so load the full pickled state.
    trainer.fit(
        model=modelmodule,
        datamodule=datamodule,
        ckpt_path=cfg.ckpt_path,
        weights_only=False,
    )
    ensemble(cfg)


if __name__ == "__main__":
    main()
