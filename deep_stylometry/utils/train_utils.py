# deep_stylometry/utils/train_utils.py

import os.path as osp
from typing import Any, Dict, Optional

import lightning as L
import psutil
import torch
import wandb
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from deep_stylometry.modules.modeling_deep_stylometry import DeepStylometry
from deep_stylometry.utils.configs.base_config import BaseConfig
from deep_stylometry.utils.data.halvest_data import HALvestDataModule
from deep_stylometry.utils.data.se_datamodule import StyleEmbeddingDataModule

NUM_PROC = psutil.cpu_count(logical=False)


def setup_datamodule(
    cfg: BaseConfig,
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
    tuning_mode: bool = False,
):
    """Use the config to set up the correct datamodule.

    Parameters
    ----------
    config: Dict[str, Any]
        Configuration dictionary containing the dataset name and other parameters.
    cache_dir: Optional[str]
        Directory to cache the dataset.
    num_proc: Optional[int]
        Number of processes to use for data loading. If None, defaults to the number
        of CPUs.

    Returns
    -------
    dm: L.LightningDataModule
        The data module for the specified dataset.
    """
    num_proc = num_proc if num_proc is not None else NUM_PROC
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
):
    """Setup the Lightning trainer with the specified configuration.

    Parameters
    ----------
    config: Dict[str, Any]
        Configuration dictionary containing the training parameters.
    logs_dir: str
        Directory to save logs.
    use_wandb: bool
        Whether to use Weights and Biases for logging.
    checkpoint_dir: Optional[str]
        Directory to save model checkpoints. If None, no checkpoints will be saved.

    Returns
    -------
    trainer: L.Trainer
        The Lightning trainer instance with the specified configuration.
    """
    # Set up callbacks
    callbacks = []

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    name = f"""{cfg.model.base_model_name}-{cfg.data.ds_name}
        -pooling:{cfg.model.pooling_method}-softmax:{cfg.model.use_softmax}
        -dist:{cfg.model.distance_weightning}"""

    # Early stopping if configured
    if cfg.execution.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=cfg.execution.early_stopping_metric,
            mode=cfg.execution.early_stopping_mode,
            patience=cfg.execution.early_stopping_patience,
        )
        callbacks.append(early_stop_callback)

    # Model checkpoint callback if checkpoint_dir is provided
    if checkpoint_dir is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=osp.join(checkpoint_dir, name),
            filename="{epoch}",
            monitor=cfg.execution.checkpoint_metric,
            mode=cfg.execution.checkpoint_mode,
            save_top_k=cfg.execution.save_top_k,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # Configure loggers
    loggers = []
    if cfg.execution.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            name=name,
            log_model=cfg.execution.log_model,
            group=cfg.group_name,
        )
        watch = cfg.execution.watch
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

    trainer = L.Trainer(
        accelerator=cfg.execution.device,
        strategy=cfg.execution.strategy,
        devices=cfg.execution.num_devices,
        max_steps=cfg.execution.max_steps,
        max_epochs=cfg.execution.max_epochs,
        val_check_interval=cfg.execution.val_check_interval,
        enable_checkpointing=checkpoint_dir is not None,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.execution.log_every_n_steps,
        accumulate_grad_batches=cfg.execution.accumulate_grad_batches,
        gradient_clip_val=cfg.execution.gradient_clip_val,
        precision=cfg.execution.precision,
    )
    return trainer


def train_tune(
    config: Dict[str, Any],
    logs_dir: str,
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
):
    """Launch hyper-parameter tuning using Ray Tune and PyTorch Lightning.

    Parameters
    ----------
    config: Dict[str, Any]
        Configuration dictionary containing the tuning parameters.
    base_config: Dict[str, Any]
        Base configuration dictionary containing the model and training parameters.
    cache_dir: Optional[str]
        Directory to cache the dataset.
    num_proc: Optional[int]
        Number of processes to use for data loading. If None, defaults to the number
        of CPUs.
    """

    cfg = BaseConfig(mode="tune").from_dict(config)

    dm = setup_datamodule(
        cfg,
        cache_dir=cache_dir,
        num_proc=num_proc,
        tuning_mode=True,
    )
    model = DeepStylometry(cfg)
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(
        TuneReportCheckpointCallback(
            {
                "val_auroc": "val_auroc",
                "val_mrr": "val_mrr",
                "val_total_loss": "val_total_loss",
                "completed_epoch": "completed_epoch",
            },
            on="validation_end",
            save_checkpoints=False,
        )
    )
    # Configure loggers
    loggers = []
    loggers.append(
        CSVLogger(
            save_dir=logs_dir,
            name=f"""tune-{cfg.model.base_model_name}-{cfg.data.ds_name}
                -pooling:{cfg.model.pooling_method}-softmax:{cfg.model.use_softmax}
                -dist:{cfg.model.distance_weightning}""",
        )
    )
    if cfg.execution.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            group=cfg.group_name,
            prefix="trial",
            log_model=False,
        )
        loggers.append(wandb_logger)

    trainer = L.Trainer(
        accelerator=cfg.tune.device,
        devices=cfg.tune.num_devices_per_trial,
        max_steps=cfg.tune.max_steps,
        max_epochs=cfg.tune.max_epochs,
        val_check_interval=None,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=loggers,
        log_every_n_steps=cfg.tune.log_every_n_steps,
        accumulate_grad_batches=cfg.tune.accumulate_grad_batches,
        gradient_clip_val=cfg.tune.gradient_clip_val,
        precision=cfg.tune.precision,
    )
    trainer.fit(model=model, datamodule=dm)

    if cfg.execution.use_wandb:
        wandb.finish()
