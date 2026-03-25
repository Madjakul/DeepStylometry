# test.py

import logging
import os

import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from deep_stylometry.callbacks import TestEvalCallback
from deep_stylometry.modules import DeepStylometry
from deep_stylometry.utils import train_utils
from deep_stylometry.utils.argparsers import TestArgparse
from deep_stylometry.utils.configs import BaseConfig
from deep_stylometry.utils.helpers import resolve_lightning_precision, set_seed
from deep_stylometry.utils.logger import logging_config

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

set_seed()
logging_config()

if __name__ == "__main__":
    args = TestArgparse.parse_known_args()
    cfg = BaseConfig(mode="test").from_yaml(args.config_path)

    # CLI overrides (allow per-job customisation without editing YAML)
    if args.test_subset is not None:
        cfg.data.test_subset = args.test_subset
    if args.ds_name is not None:
        cfg.data.ds_name = args.ds_name

    logging.info("Preparing data module...")
    dm = train_utils.setup_datamodule(
        cfg=cfg,
        processed_ds_dir=args.processed_ds_dir,
        num_proc=args.num_proc,
        cache_dir=args.cache_dir,
    )

    if cfg.model.pooling_method == "pli":
        patch_tag = f"pli-{cfg.model.patch_method}-n{cfg.model.patch_size}"
    else:
        patch_tag = cfg.model.pooling_method
    name = (
        f"test__{cfg.model.base_checkpoint}__{cfg.data.ds_name}"
        f"__pooling-{patch_tag}__{cfg.data.test_subset}"
        f"__skip_list-{cfg.model.skip_list}"
    ).replace("/", "-").lower()
    loggers = []
    if cfg.test.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            name=name,
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
