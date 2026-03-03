# test.py

# test.py

import argparse
import logging
import os

import lightning as L

from deep_stylometry.modules import DeepStylometry
from deep_stylometry.utils import train_utils
from deep_stylometry.utils.configs import BaseConfig
from deep_stylometry.utils.helpers import set_seed
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
    args = parser.parse_args()

    cfg = BaseConfig(mode="test").from_yaml(args.config_path)

    logging.info("Preparing data module...")
    dm = train_utils.setup_datamodule(
        cfg=cfg,
        processed_ds_dir=args.processed_ds_dir,
        num_proc=args.num_proc,
    )

    logging.info(f"Loading model from {args.checkpoint_path}...")
    # Load weights from your saved checkpoint
    model = DeepStylometry.load_from_checkpoint(args.checkpoint_path, cfg=cfg)

    # Force 1 GPU for testing to avoid distributed gather complications
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,  # Set to your WandbLogger/CSVLogger if you want to save the test metrics remotely
    )

    logging.info("=== Starting Evaluation ===")
    trainer.test(model=model, datamodule=dm)
    logging.info("=== Evaluation Finished ===")
