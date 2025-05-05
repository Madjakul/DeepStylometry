# tune.py

import logging

import psutil

from deep_stylometry.utils import tune_utils
from deep_stylometry.utils.argparsers import TuneArgparse
from deep_stylometry.utils.helpers import load_config_from_file
from deep_stylometry.utils.logger import logging_config

logging_config()
NUM_PROC = psutil.cpu_count(logical=False) - 1


if __name__ == "__main__":
    args = TuneArgparse.parse_known_args()
    config = load_config_from_file(args.config_path)

    tuner = tune_utils.setup_tuner(
        config=config,
        ray_storage_path=args.ray_storage_path,
        use_wandb=config["use_wandb"],
        cache_dir=args.cache_dir,
        num_proc=args.num_proc if args.num_proc is not None else NUM_PROC,
    )
    results = tuner.fit()
    logging.info("--- Tuning finished ---")
