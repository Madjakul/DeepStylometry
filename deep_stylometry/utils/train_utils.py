# deep_stylometry/utils/train_utils.py

from typing import Any, Dict, Optional

import lightning as L
import psutil
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from deep_stylometry.modules import DeepStylometry
from deep_stylometry.utils.data.halvest_data import HALvestDataModule
from deep_stylometry.utils.data.se_data import SEDataModule

NUM_PROC = psutil.cpu_count(logical=False)


def setup_datamodule(
    config: Dict[str, Any],
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
):
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
        ds_name=config["ds_name"],
        config_name=config.get("config_name", None),
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
        do_late_interaction=config.get("do_late_interaction", False),
        use_max=config.get("initial_gumbel_temp", None) is not None,
        initial_gumbel_temp=config.get("initial_gumbel_temp", None),
        auto_anneal_gumbel=config.get("auto_anneal_gumbel", True),
        gumbel_temp_annealing_rate=config.get("gumbel_temp_annealing_rate", 1e-3),
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
    # cache_dir: Optional[str] = None,
    # num_proc: Optional[int] = None,
):
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
            save_top_k=config.get("save_top_k", 3),
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

    # Set up trainer
    trainer = L.Trainer(
        accelerator=config.get("device", "cpu"),
        devices=config.get("num_devices", -1),
        max_epochs=config.get("max_epochs", 3),
        enable_checkpointing=checkpoint_dir is not None,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.get("log_every_n_steps", 1),
        accumulate_grad_batches=config.get("accumulate_grad_batches", 2),
        gradient_clip_val=config.get("gradient_clip_val", 1e-3),
        precision=config.get("precision", "16-mixed"),
    )

    # dm = setup_datamodule(config=config, cache_dir=cache_dir, num_proc=num_proc)
    # model = setup_model(config=config["model"])
    # trainer.fit(model, datamodule=dm)
    # if config.get("run_test", False):
    #     trainer.test(model, datamodule=dm)
    return trainer


def train_tune(
    config: Dict[str, Any],
    base_config: Dict[str, Any],
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
):
    merged_config = base_config.copy()
    merged_config.update(config)

    dm = setup_datamodule(
        merged_config,
        cache_dir=cache_dir,
        num_proc=num_proc,
    )
    model = setup_model(config)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(
        accelerator=merged_config.get("device", "cpu"),
        devices=merged_config.get("num_devices", -1),
        max_epochs=merged_config.get("max_epochs", 3),
        callbacks=[lr_monitor],
        enable_checkpointing=False,
        log_every_n_steps=merged_config.get("log_every_n_steps", 1),
        accumulate_grad_batches=merged_config["accumulate_grad_batches"],
        gradient_clip_val=merged_config["gradient_clip_val"],
        precision=merged_config.get("precision", "16-mixed"),
    )
    trainer.fit(model=model, datamodule=dm)
