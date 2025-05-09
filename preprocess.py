# preprocess.py

import logging

import psutil

from deep_stylometry.utils import helpers, logging_config
from deep_stylometry.utils.argparsers import PreprocessArgparse
from deep_stylometry.utils.data import preprocessing

NUM_PROC = psutil.cpu_count(logical=False) - 1
logging_config()


if __name__ == "__main__":
    args = PreprocessArgparse.parse_known_args()
    config = helpers.load_config_from_file(args.config_path)

    logging.info(f"{('=' * helpers.WIDTH)}")
    logging.info(f"Preprocessing HALvest".center(helpers.WIDTH))
    logging.info(f"{('=' * helpers.WIDTH)}")

    preprocessing.run(
        target_ds_name=config["target_ds_name"],
        batch_size=config["batch_size"],
        num_proc=args.num_proc if args.num_proc else NUM_PROC,
        load_from_cache_file=config["load_from_cache_file"],
        cache_dir=args.cache_dir,
    )
