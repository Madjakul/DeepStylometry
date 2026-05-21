# deep_stylometry/utils/train_utils.py

import os
import os.path as osp
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from deep_stylometry.callbacks import (EvalRuntimeMonitor,
                                       LogarithmicValidationCallback,
                                       LossVarianceMonitor)
from deep_stylometry.utils.configs.base_config import BaseConfig
from deep_stylometry.utils.data.halvest_datamodule import \
    HALvestContrastiveDatamodule
from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
from deep_stylometry.utils.helpers import resolve_lightning_precision


def setup_datamodule(
    cfg: BaseConfig,
    processed_ds_dir: str,
    num_proc: int,
    cache_dir: Optional[str] = None,
) -> L.LightningDataModule:
    """Instantiate the appropriate datamodule for ``cfg.data.ds_name``."""
    dm_map = {
        "halvest": HALvestContrastiveDatamodule,
        "pan19": PAN19Datamodule,
    }

    kwargs: dict = dict(
        cfg=cfg,
        processed_ds_dir=processed_ds_dir,
        num_proc=num_proc,
        cache_dir=cache_dir,
    )
    if cfg.data.ds_name == "pan19":
        kwargs["pan19_zip"] = os.environ.get("PAN19_ZIP")

    dm = dm_map[cfg.data.ds_name](**kwargs)
    return dm


def setup_trainer(
    cfg: BaseConfig,
    model: torch.nn.Module,
    logs_dir: str,
    checkpoint_dir: Optional[str] = None,
) -> L.Trainer:
    """Build and return a configured Lightning Trainer.

    Attaches callbacks (LR monitor, loss variance monitor, eval runtime
    monitor, logarithmic validation), optional WandB and CSV loggers, and
    a ModelCheckpoint when ``checkpoint_dir`` is provided.

    Parameters
    ----------
    cfg : BaseConfig
        Global configuration object.
    model : torch.nn.Module
        The model to optionally watch with WandB.
    logs_dir : str
        Root directory for CSV logs.
    checkpoint_dir : str, optional
        If given, checkpoints are saved under
        ``checkpoint_dir/<experiment_name>/``.

    Returns
    -------
    L.Trainer
        A fully configured Lightning Trainer.
    """
    # Set up callbacks
    callbacks = []

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    callbacks.append(LossVarianceMonitor(window_size=100))
    callbacks.append(EvalRuntimeMonitor())
    callbacks.append(
        LogarithmicValidationCallback(start_step=10, growth=1.5, max_interval=1000)
    )

    if cfg.model.pooling_method == "pli":
        patch_tag = f"{cfg.model.patch_method}-n{cfg.model.patch_size}"
        if cfg.model.patch_compression != "mean":
            patch_tag += f"-{cfg.model.patch_compression}"
        name = (
            f"{cfg.model.base_checkpoint}__{cfg.data.ds_name}"
            f"__pooling-pli-{patch_tag}__skip_list-{cfg.model.skip_list}"
        ).replace("/", "-").lower()
    else:
        name = (
            f"{cfg.model.base_checkpoint}__{cfg.data.ds_name}"
            f"__pooling-{cfg.model.pooling_method}__skip_list-{cfg.model.skip_list}"
        ).replace("/", "-").lower()

    # Model checkpoint callback if checkpoint_dir is provided
    if checkpoint_dir is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=osp.join(checkpoint_dir, name),
            filename="step-{step}",
            save_top_k=-1,
            monitor=None,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # Configure loggers
    loggers = []
    if cfg.train.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            name=name,
            log_model=cfg.train.log_model,
        )
        watch = cfg.train.watch
        if watch is not None:
            wandb_logger.watch(
                model=model,
                log=watch,
                log_graph=False,
                log_freq=cfg.train.accumulate_grad_batches * 1000,
            )
        loggers.append(wandb_logger)

    # Add CSV logger by default
    csv_logger = CSVLogger(save_dir=logs_dir, name=name)
    loggers.append(csv_logger)

    precision, _ = resolve_lightning_precision(cfg.train.precision)

    trainer = L.Trainer(
        accelerator=cfg.train.device,
        strategy=cfg.train.strategy,
        devices=cfg.train.num_devices,
        max_steps=cfg.train.max_steps,
        max_epochs=cfg.train.max_epochs,
        num_sanity_val_steps=0,
        val_check_interval=100_000_000,
        check_val_every_n_epoch=None,
        enable_checkpointing=checkpoint_dir is not None,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.train.log_every_n_steps,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        precision=precision,
        overfit_batches=cfg.train.overfit_batches,
    )
    return trainer
