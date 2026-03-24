# tune.py — Entry point for PLI hyperparameter search via Optuna

import logging
import os

from deep_stylometry.experiments.optuna_pli_search import run_search
from deep_stylometry.utils.argparsers import TuneArgparse
from deep_stylometry.utils.logger import logging_config

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging_config()

if __name__ == "__main__":
    args = TuneArgparse.parse_known_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    logging.info(f"Starting PLI hyperparameter search: {args.n_trials} trials, "
                 f"{args.max_steps} steps each")
    logging.info(f"Base config: {args.config_path}")
    logging.info(f"Best params will be written to: {args.output_path}")

    run_search(
        config_path=args.config_path,
        processed_ds_dir=args.processed_ds_dir,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        output_path=args.output_path,
        num_proc=args.num_proc,
        study_name=args.study_name,
        storage=args.storage,
    )
