# train.py

import logging
import os

import psutil

from deep_stylometry.modules import DeepStylometry, StyleEmbedding
from deep_stylometry.utils import train_utils
from deep_stylometry.utils.argparsers import TrainArgparse
from deep_stylometry.utils.helpers import load_config_from_file
from deep_stylometry.utils.logger import logging_config

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_PROC = psutil.cpu_count(logical=False)

logging_config()


if __name__ == "__main__":
    args = TrainArgparse.parse_known_args()
    config = load_config_from_file(args.config_path)

    logging.info("Preparing data module...")
    dm = train_utils.setup_datamodule(
        config=config,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc if args.num_proc is not None else NUM_PROC,
    )

    if config.get("do_test", False):
        logging.info("Setting up DataModule for test split...")
        dm.setup("test")

    model = None
    arch = config.get("architecture", "").lower()

    # Training
    if config.get("do_train", False):
        logging.info("--- Fine-tuning (training) ---")

        # Instantiate the correct model for training
        if arch == "deep-stylometry":
            logging.info("Instantiating DeepStylometry model for training.")
            model = train_utils.setup_model(config=config)
        else:
            logging.info("Instantiating StyleEmbedding model for training.")
            model = StyleEmbedding()

        # Create the trainer (with checkpointing, wandb, etc.)
        trainer = train_utils.setup_trainer(
            config=config,
            model=model,
            testing_mode=False,
            logs_dir=args.logs_dir,
            use_wandb=config.get("use_wandb"),
            checkpoint_dir=args.checkpoint_dir,
        )

        trainer.fit(model=model, datamodule=dm)
        logging.info("--- Fine-tuning finished ---")

    # Testing
    if config.get("do_test", False):
        logging.info("--- Testing ---")

        # If we just trained a DeepStylometry model, reuse it.
        # Otherwise, if no model exists yet, we need to load or instantiate for testing.
        if model is None:
            if arch == "deep-stylometry":
                # Must have provided a checkpoint_path when testing from scratch
                if args.checkpoint_path:
                    logging.info(
                        "Loading DeepStylometry model from checkpoint for testing."
                    )
                    model = DeepStylometry.load_from_checkpoint(args.checkpoint_path)
                else:
                    raise ValueError(
                        """Cannot test DeepStylometry without"""
                        """ --checkpoint_path=<path_to_checkpoint>"""
                    )
            else:
                logging.info("Instantiating StyleEmbedding model for testing.")
                model = StyleEmbedding()

        test_config = {
            **config,
            "num_devices": 1,
            "strategy": "auto",
            "max_epochs": 1,
            "accumulate_grad_batches": 1,
        }

        test_trainer = train_utils.setup_trainer(
            config=test_config,
            model=model,
            testing_mode=True,
            logs_dir=args.logs_dir,
            use_wandb=config.get("use_wandb"),
            checkpoint_dir=None,
        )

        test_trainer.test(model=model, datamodule=dm)
        logging.info("--- Testing finished ---")
