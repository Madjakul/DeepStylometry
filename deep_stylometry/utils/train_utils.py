# deep_stylometry/utils/train_utils.py

from typing import Any, Dict, Optional

import lightning as L
import psutil
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from deep_stylometry.modules import DeepStylometry
from deep_stylometry.utils.data.halvest_data import HALvestDataModule
from deep_stylometry.utils.data.se_data import SEDataModule

NUM_PROC = psutil.cpu_count(logical=False)


def setup_datamodule(
    config: Dict[str, Any],
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
        Number of processes to use for data loading. If None, defaults to the number of CPUs.

    Returns
    -------
    dm: L.LightningDataModule
        The data module for the specified dataset.
    """
    num_proc = num_proc if num_proc is not None else NUM_PROC
    dm_map = {"se": SEDataModule, "halvest": HALvestDataModule}

    dm = dm_map[config["ds_name"]](
        batch_size=config["batch_size"],
        num_proc=num_proc,
        tokenizer_name=config["tokenizer_name"],
        max_length=config["max_length"],
        map_batch_size=config["map_batch_size"],
        load_from_cache_file=config["load_from_cache_file"],
        cache_dir=cache_dir,
        config_name=config.get("config_name", None),
        mlm_collator=config.get("mlm_collator", False),
        tuning_mode=tuning_mode,
    )
    return dm


def setup_model(config: Dict[str, Any]):
    """Use the config to set up the model.

    Parameters
    ----------
    config: Dict[str, Any]
        Configuration dictionary containing the model parameters.

    Returns
    -------
    model: DeepStylometry
        The model instance with the specified parameters.
    """
    model = DeepStylometry(
        optim_name=config.get("optim_name", "adamw"),
        base_model_name=config["base_model_name"],
        batch_size=config["batch_size"],
        seq_len=config["max_length"],
        is_decoder_model=config["is_decoder_model"],
        lr=config.get("lr", 2e-5),
        dropout=config.get("dropout", 0.1),
        weight_decay=config.get("weight_decay", 1e-2),
        lm_weight=config.get("lm_weight", 1.0),
        contrastive_weight=config.get("contrastive_weight", 1.0),
        contrastive_temp=config.get("contrastive_temp", 7e-2),
        do_late_interaction=config.get("do_late_interaction", False),
        use_max=config.get("initial_gumbel_temp", None) is None,
        initial_gumbel_temp=config.get("initial_gumbel_temp", 1.0),
        auto_anneal_gumbel=config.get("auto_anneal_gumbel", True),
        gumbel_linear_delta=config.get("gumbel_linear_delta", 1e-3),
        min_gumbel_temp=config.get("min_gumbel_temp", 1e-9),
        do_distance=config.get("do_distance", True),
        exp_decay=config.get("exp_decay", False),
        alpha=config.get("alpha", 0.5),
        project_up=config.get("project_up", None),
    )

    return model


def setup_trainer(
    config: Dict[str, Any],
    logs_dir: str,
    use_wandb: bool = False,
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

    # Early stopping if configured
    if config.get("early_stopping", False):
        early_stop_callback = EarlyStopping(
            monitor=config.get("early_stopping_metric", "val_total_loss"),
            mode=config.get("early_stopping_mode", "min"),
            patience=config.get("early_stopping_patience", 5),
        )
        callbacks.append(early_stop_callback)

    # Model checkpoint callback if checkpoint_dir is provided
    if checkpoint_dir is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_auroc:.4f}",
            monitor=config.get("checkpoint_metric", "val_auroc"),
            mode=config.get("checkpoint_mode", "max"),
            save_top_k=config.get("save_top_k", 2),
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # Configure loggers
    loggers = []
    if use_wandb:
        wandb_logger = WandbLogger(
            project=config.get("project_name", "deep_stylometry"),
            name=config.get("experiment_name", "training_run"),
            log_model=config.get("log_model", False),
        )
        loggers.append(wandb_logger)

    # Add CSV logger by default
    csv_logger = CSVLogger(
        save_dir=logs_dir,
        name=config.get("experiment_name", "training_run"),
    )
    loggers.append(csv_logger)

    trainer = L.Trainer(
        accelerator=config.get("device", "cpu"),
        devices=config.get("num_devices", -1),
        max_epochs=config.get("max_epochs", 3),
        enable_checkpointing=checkpoint_dir is not None,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=config.get("log_every_n_steps", 10),
        accumulate_grad_batches=config["accumulate_grad_batches"],
        gradient_clip_val=config["gradient_clip_val"],
        precision=config.get("precision", "16-mixed"),
    )
    return trainer


def train_tune(
    config: Dict[str, Any],
    base_config: Dict[str, Any],
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
        Number of processes to use for data loading. If None, defaults to the number of CPUs.
    """
    merged_config = base_config.copy()
    merged_config.update(config)

    dm = setup_datamodule(
        merged_config,
        cache_dir=cache_dir,
        num_proc=num_proc,
        tuning_mode=True,
    )
    model = setup_model(merged_config)
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(
        TuneReportCheckpointCallback(
            {
                "loss": "val_total_loss",
                "auroc": "val_auroc",
                "f1": "val_f1",
                "precision": "val_precision",
                "recall": "val_recall",
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
            name=merged_config.get("experiment_name", "training-run"),
        )
    )
    if merged_config["use_wandb"]:
        wandb_logger = WandbLogger(
            project=merged_config.get("project_name", "deep-stylometry"),
            group="tune",
            prefix="trial",
            log_model=False,
        )
        loggers.append(wandb_logger)

    trainer = L.Trainer(
        accelerator=merged_config.get("device", "cpu"),
        devices=merged_config.get("num_devices_per_trial", -1),
        max_steps=1250,
        max_epochs=-1,
        val_check_interval=625,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=loggers,
        log_every_n_steps=merged_config.get("log_every_n_steps", 1),
        accumulate_grad_batches=merged_config.get("accumulate_grad_batches", 2),
        gradient_clip_val=merged_config.get("gradient_clip_val", 1e-3),
        precision=merged_config.get("precision", "16-mixed"),
    )
    trainer.fit(model=model, datamodule=dm)
