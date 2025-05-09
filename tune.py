# tune.py

import logging
import os

import psutil
import wandb

from deep_stylometry.utils import tune_utils
from deep_stylometry.utils.argparsers import TuneArgparse
from deep_stylometry.utils.helpers import load_config_from_file
from deep_stylometry.utils.logger import logging_config

# os.environ["RAY_memory_monitor_refresh_ms"] = "0"
logging_config()
NUM_PROC = psutil.cpu_count(logical=False) - 1


if __name__ == "__main__":
    args = TuneArgparse.parse_known_args()
    config = load_config_from_file(args.config_path)
    logging.info(f"--- Tuning hyperparameters ---")
    logging.info(f"Config file: {args.config_path}")

    if config["use_wandb"]:
        wandb.init(
            project=config["project_name"],
            name=config["experiment_name"],
            group="tuning",
            config=config,
            dir=args.cache_dir,
        )

    tuner = tune_utils.setup_tuner(
        config=config,
        ray_storage_path=args.ray_storage_path,
        use_wandb=config["use_wandb"],
        cache_dir=args.cache_dir,
        num_proc=args.num_proc if args.num_proc is not None else NUM_PROC,
    )
    results = tuner.fit()
    logging.info("--- Tuning finished ---")
