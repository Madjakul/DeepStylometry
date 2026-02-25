# deep_stylometry/utils/train_utils.py

import os.path as osp
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from deep_stylometry.callbacks import (
    EvalRuntimeMonitor,
    GradNormMonitor,
    LogarithmicValidationCallback,
    LossVarianceMonitor,
    RetrievalEvalCallback,
)
from deep_stylometry.utils.configs.base_config import BaseConfig
from deep_stylometry.utils.data.halvest_datamodule import HALvestContrastiveDatamodule
from deep_stylometry.utils.data.se_datamodule import StyleEmbeddingDatamodule
from deep_stylometry.utils.helpers import resolve_lightning_precision


def setup_datamodule(
    cfg: BaseConfig,
    processed_ds_dir: str,
    num_proc: int,
    cache_dir: Optional[str] = None,
) -> L.LightningDataModule:
    dm_map = {"se": StyleEmbeddingDatamodule, "halvest": HALvestContrastiveDatamodule}

    dm = dm_map[cfg.data.ds_name](
        cfg=cfg,
        processed_ds_dir=processed_ds_dir,
        num_proc=num_proc,
        cache_dir=cache_dir,
    )
    return dm


def setup_trainer(
    cfg: BaseConfig,
    model: torch.nn.Module,
    logs_dir: str,
    checkpoint_dir: Optional[str] = None,
) -> L.Trainer:
    # Set up callbacks
    callbacks = []

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    callbacks.append(LossVarianceMonitor(window_size=100))
    callbacks.append(GradNormMonitor())
    callbacks.append(EvalRuntimeMonitor())
    callbacks.append(
        LogarithmicValidationCallback(start_step=1, growth=1.5, max_interval=1000)
    )
    callbacks.append(RetrievalEvalCallback())

    name = (
        f"{cfg.model.base_checkpoint}__{cfg.data.ds_name}"
        f"__pooling-{cfg.model.pooling_method}"
    ).replace("/", "-")

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
            log_model=cfg.execution.log_model,  # type: ignore
        )
        watch = cfg.execution.watch  # type: ignore
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

    precision, _ = resolve_lightning_precision(cfg.execution.precision)  # type: ignore

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
