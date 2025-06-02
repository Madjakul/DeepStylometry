# train.py

import logging
import os

import psutil

from deep_stylometry.modules import DeepStylometry
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

    logging.info("Preparing data...")
    dm = train_utils.setup_datamodule(
        config=config,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc if args.num_proc is not None else NUM_PROC,
    )

    if config["do_test"]:
        logging.info("Setting up test data module...")
        dm.setup("test")

    if config["do_train"]:
        logging.info(f"--- Fine-tuning ---")
        model = train_utils.setup_model(config=config)

        trainer = train_utils.setup_trainer(
            config=config,
            model=model,
            logs_dir=args.logs_dir,
            use_wandb=config.get("use_wandb"),
            checkpoint_dir=args.checkpoint_dir,
        )
        trainer.fit(model=model, datamodule=dm)
        logging.info("--- Fine-tuning finished ---")

    if config["do_test"]:
        logging.info("--- Testing ---")
        # Check if 'model' is defined in the current scope; if not, load from checkpoint
        if "model" not in locals() and args.checkpoint_path is not None:
            model = DeepStylometry.load_from_checkpoint(args.checkpoint_path)
        elif "model" not in locals():
            raise ValueError(
                "Model is not accessible, and no 'test_checkpoint_path' was provided."
            )

        test_trainer = train_utils.setup_trainer(
            config={
                **config,
                "num_devices": 1,
                "strategy": "auto",
                "max_epochs": 1,
                "accumulate_grad_batches": 1,
            },
            model=model,
            logs_dir=args.logs_dir,
            use_wandb=config.get("use_wandb"),
            checkpoint_dir=None,  # No checkpointing needed during testing
        )
        test_trainer.test(model=model, datamodule=dm)
        logging.info("--- Testing finished ---")
