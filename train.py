# train.py

import logging
import os

import psutil

from deep_stylometry.modules import DeepStylometry
from deep_stylometry.utils import train_utils
from deep_stylometry.utils.argparsers import TrainArgparse
from deep_stylometry.utils.configs import BaseConfig
from deep_stylometry.utils.logger import logging_config

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_PROC = psutil.cpu_count(logical=False)

logging_config()


if __name__ == "__main__":
    args = TrainArgparse.parse_known_args()
    cfg = BaseConfig(mode="train").from_yaml(args.config_path)

    logging.info("Preparing data module...")
    dm = train_utils.setup_datamodule(
        cfg=cfg,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc if args.num_proc is not None else NUM_PROC,
    )

    if cfg.do_test:
        logging.info("Setting up DataModule for test split...")
        dm.setup("test")

    model = None
    trainer = None

    # Training
    if cfg.do_train:
        logging.info("--- Fine-tuning ---")
        model = DeepStylometry(cfg)

        # Create the trainer (with checkpointing, wandb, etc.)
        trainer = train_utils.setup_trainer(
            cfg=cfg,
            model=model,
            logs_dir=args.logs_dir,
            checkpoint_dir=args.checkpoint_dir,
        )

        trainer.fit(model=model, datamodule=dm)
        logging.info("--- Fine-tuning finished ---")

    # Testing
    if cfg.do_test:
        logging.info("--- Testing ---")

        # If we just trained a DeepStylometry model, reuse it.
        # Otherwise, if no model exists yet, we need to load or instantiate for testing.
        if trainer is None:
            # Must have provided a checkpoint_path when testing from scratch
            if args.checkpoint_path is not None:
                logging.info("Loading DeepStylometry model from checkpoint.")
                model = DeepStylometry.load_from_checkpoint(args.checkpoint_path)
            else:
                model = DeepStylometry(cfg)

            trainer = train_utils.setup_trainer(
                cfg=cfg,
                model=model,
                logs_dir=args.logs_dir,
                checkpoint_dir=None,
            )

        trainer.test(model=model, datamodule=dm)
        logging.info("--- Testing finished ---")
