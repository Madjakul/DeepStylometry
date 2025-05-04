# deep_stylometry/utils/train_utils.py

import os
from typing import Any, Dict, Optional

import lightning as L
import psutil
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from deep_stylometry.modules import DeepStylometry
from deep_stylometry.utils.data.halvest_data import HALvestDataModule
from deep_stylometry.utils.data.se_data import SEDataModule

NUM_PROC = psutil.cpu_count(logical=False)


def setup_datamodule(
    dm_config: Dict[str, Any],
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
):
    num_proc = num_proc if num_proc is not None else NUM_PROC
    dm_map = {"se": SEDataModule, "halvest": HALvestDataModule}

    dm = dm_map[dm_config["ds_name"]](
        batch_size=dm_config["batch_size"],
        num_proc=num_proc,
        tokenizer_name=dm_config["tokenizer_name"],
        max_length=dm_config["max_length"],
        map_batch_size=dm_config["map_batch_size"],
        load_from_cache_file=dm_config["load_from_cache_file"],
        cache_dir=cache_dir,
        ds_name=dm_config["ds_name"],
        config_name=dm_config.get("config_name", None),
    )
    return dm


def setup_model(config: Dict[str, Any]):

    model = DeepStylometry(
        optim_name=config["optim_name"],
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
        do_late_interaction=config["late_interaction"].get("do_late_interaction", True),
        use_max=config["late_interaction"].get("use_max", False),
        initial_gumbel_temp=config["late_interaction"].get("initial_gumbel_temp", 1.0),
        auto_anneal_gumbel=config["late_interaction"].get("auto_anneal_gumbel", True),
        gumbel_temp_annealing_rate=config["late_interaction"].get(
            "gumbel_temp_annealing_rate", 1e-3
        ),
        min_gumbel_temp=config["late_interaction"].get("min_gumbel_temp", 1e-9),
        do_distance=config["late_interaction"].get("do_distance", True),
        exp_decay=config["late_interaction"]["exp_decay"],
        alpha=config["late_interaction"].get("alpha", 0.5),
        project_up=config.get("project_up", None),
    )

    return model


def train(
    config: Dict[str, Any],
    device: str,
    use_wandb: bool = False,
    checkpoint_dir: bool = False,
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
):
    # Set up callbacks
    callbacks = []

    # Learning rate monitor
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Early stopping if configured
    if config.get("early_stopping", False):
        early_stop_callback = L.pytorch.callbacks.EarlyStopping(
            monitor=config.get("early_stopping_metric", "val_total_loss"),
            mode=config.get("early_stopping_mode", "min"),
            patience=config.get("early_stopping_patience", 3),
        )
        callbacks.append(early_stop_callback)

    # Model checkpoint callback if checkpoint_dir is provided
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_auroc:.4f}",
            monitor=config.get("checkpoint_metric", "val_auroc"),
            mode=config.get("checkpoint_mode", "max"),
            save_top_k=config.get("save_top_k", 3),
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # Configure loggers
    loggers = []
    if use_wandb:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            project=project_name or config.get("project_name", "deep_stylometry"),
            name=experiment_name or config.get("experiment_name", "training_run"),
            log_model=config.get("log_model", False),
        )
        loggers.append(wandb_logger)

    # Add CSV logger by default
    csv_logger = L.pytorch.loggers.CSVLogger(
        save_dir=checkpoint_dir or "logs",
        name=experiment_name or config.get("experiment_name", "training_run"),
    )
    loggers.append(csv_logger)

    # Set up trainer
    trainer = L.Trainer(
        accelerator=config.get("accelerator", "gpu"),
        devices=config.get("devices", -1),
        max_epochs=config.get("max_epochs", 200),
        enable_checkpointing=checkpoint_dir is not None,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.get("log_every_n_steps", 1),
        accumulate_grad_batches=config.get("accumulate_grad_batches", 2),
        gradient_clip_val=config.get("gradient_clip_val", 1e-3),
        precision=config.get("precision", "16-mixed"),
    )

    # Set up data module
    dm = setup_datamodule(
        dm_config=config["data"],
        cache_dir=cache_dir,
        num_proc=num_proc,
    )

    # Set up model
    model = setup_model(
        model_config=config["model"],
        dm_config=config["data"],
    )

    # Train model
    trainer.fit(model, datamodule=dm)

    # Test model if configured
    test_results = None
    if config.get("run_test", False):
        test_results = trainer.test(model, datamodule=dm)


def train_tune(
    config: Dict[str, Any],
    base_config: Dict[str, Any],
    device: str,
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
):
    merged_config = base_config.copy()
    for key, value in config.items():
        if key in merged_config["model"]:
            merged_config["model"][key] = value
        elif key in merged_config["model"]["late_interaction"]:
            merged_config["model"]["late_interaction"][key] = value
        elif key in merged_config["data"]:
            merged_config["data"][key] = value
    dm = setup_datamodule(
        config["datamodule"],
        cache_dir=cache_dir,
        num_proc=num_proc,
    )
    model = setup_model(config["model"])
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(
        accelerator=device,
        devices=-1,
        max_epochs=merged_config.get("max_epochs", 3),
        callbacks=[lr_monitor],
        enable_checkpointing=False,
        log_every_n_steps=merged_config.get("log_every_n_steps", 1),
        accumulate_grad_batches=config["accumulate_grad_batches"],
        gradient_clip_val=config["gradient_clip_val"],
        precision=merged_config.get("precision", "16-mixed"),
    )
    trainer.fit(model=model, datamodule=dm)
