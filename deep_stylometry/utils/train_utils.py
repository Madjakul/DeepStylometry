# deep_stylometry/utils/train_utils.py

import os.path as osp
from typing import Any, Dict, Optional

import lightning as L
import psutil
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from deep_stylometry.utils.configs.base_config import BaseConfig
from deep_stylometry.utils.data.halvest_datamodule import HALvestDataModule
from deep_stylometry.utils.data.se_datamodule import StyleEmbeddingDataModule
from deep_stylometry.utils.helpers import resolve_lightning_precision

NUM_PROC = psutil.cpu_count(logical=False)


def setup_datamodule(
    cfg: BaseConfig,
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
    tuning_mode: bool = False,
) -> L.LightningDataModule:
    dm_map = {"se": StyleEmbeddingDataModule, "halvest": HALvestDataModule}

    dm = dm_map[cfg.data.ds_name](
        batch_size=cfg.data.batch_size,
        num_proc=num_proc if num_proc is not None else NUM_PROC,
        tokenizer_name=cfg.data.tokenizer_name,
        max_length=cfg.data.max_length,
        map_batch_size=cfg.data.map_batch_size,
        load_from_cache_file=cfg.data.load_from_cache_file,
        cache_dir=cache_dir,
        config_name=cfg.data.config_name,
        mlm_collator=cfg.data.mlm_collator,
        tuning_mode=tuning_mode,
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

    name = (
        f"{cfg.model.base_model_name}-{cfg.data.ds_name}"
        f"-pooling:{cfg.model.pooling_method}-softmax:{cfg.model.use_softmax}"
        f"-gumbel:{cfg.model.initial_gumbel_temp}-dist:{cfg.model.distance_weightning}"
    ).replace("/", "-")

    # Model checkpoint callback if checkpoint_dir is provided
    if checkpoint_dir is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=osp.join(checkpoint_dir, name),
            filename="{epoch}",
            monitor=cfg.execution.checkpoint_metric,  # type: ignore
            mode=cfg.execution.checkpoint_mode,  # type: ignore
            save_top_k=cfg.execution.save_top_k,  # type: ignore
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # Configure loggers
    loggers = []
    if cfg.execution.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            name=name,
            log_model=cfg.execution.log_model,  # type: ignore
            group=cfg.group_name,
        )
        watch = cfg.execution.watch  # type: ignore
        if watch is not None:
            wandb_logger.watch(
                model=model,
                log=watch,
                log_graph=False,
                log_freq=cfg.execution.accumulate_grad_batches * 100,
            )
        loggers.append(wandb_logger)

    # Add CSV logger by default
    csv_logger = CSVLogger(save_dir=logs_dir, name=name)
    loggers.append(csv_logger)

    if cfg.mode == "train":
        if cfg.execution.strategy.startswith("ddp"):  # type: ignore
            strategy = DDPStrategy(
                find_unused_parameters=cfg.execution.strategy.endswith(  # type: ignore
                    "find_unused_parameters_true"
                ),
                process_group_backend=cfg.execution.process_group_backend,  # type: ignore
            )
        else:
            strategy = cfg.execution.strategy  # type: ignore

    precision, _ = resolve_lightning_precision(cfg.execution.precision)  # type: ignore
    trainer = L.Trainer(
        accelerator=cfg.execution.device,
        strategy=strategy,  # type: ignore
        devices=cfg.execution.num_devices,  # type: ignore
        max_steps=cfg.execution.max_steps,
        max_epochs=cfg.execution.max_epochs,
        val_check_interval=cfg.execution.val_check_interval,  # type: ignore
        enable_checkpointing=checkpoint_dir is not None,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.execution.log_every_n_steps,
        accumulate_grad_batches=cfg.execution.accumulate_grad_batches,
        gradient_clip_val=cfg.execution.gradient_clip_val,
        precision=precision,
    )
    return trainer
