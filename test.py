# test.py

import argparse
import logging
import os

import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from deep_stylometry.callbacks import TestEvalCallback
from deep_stylometry.modules import DeepStylometry
from deep_stylometry.utils import train_utils
from deep_stylometry.utils.configs import BaseConfig
from deep_stylometry.utils.helpers import resolve_lightning_precision, set_seed
from deep_stylometry.utils.logger import logging_config

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

set_seed()
logging_config()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--processed_ds_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--logs_dir", type=str, default="logs")
    args = parser.parse_args()

    cfg = BaseConfig(mode="test").from_yaml(args.config_path)

    logging.info("Preparing data module...")
    dm = train_utils.setup_datamodule(
        cfg=cfg,
        processed_ds_dir=args.processed_ds_dir,
        num_proc=args.num_proc,
    )

    name = (
        (
            f"{cfg.model.base_checkpoint}__{cfg.data.ds_name}"
            f"__pooling-{cfg.model.pooling_method}__{cfg.data.test_subset}"
        )
        .replace("/", "_")
        .lower()
    )
    loggers = []
    if cfg.train.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            name=name,
            log_model=cfg.train.log_model,
        )
        loggers.append(wandb_logger)

    # Add CSV logger by default
    csv_logger = CSVLogger(save_dir=args.logs_dir, name=name)
    loggers.append(csv_logger)

    logging.info(f"Loading model from {args.checkpoint_path}...")
    # Load weights from your saved checkpoint
    model = DeepStylometry.load_from_checkpoint(args.checkpoint_path, cfg=cfg)
    precision, _ = resolve_lightning_precision(cfg.test.precision)

    # Force 1 GPU for testing to avoid distributed gather complications
    trainer = L.Trainer(
        accelerator=cfg.test.device,
        devices=cfg.test.num_devices,
        logger=loggers,  # Set to your WandbLogger/CSVLogger if you want to save the test metrics remotely
        callbacks=[TestEvalCallback(cfg)],
        precision=precision,
    )

    logging.info("=== Starting Evaluation ===")
    trainer.test(model=model, datamodule=dm)
    logging.info("=== Evaluation Finished ===")
