# train.py

import psutil

from deep_stylometry.utils import train_utils
from deep_stylometry.utils.argparsers import TrainArgparse
from deep_stylometry.utils.helpers import load_config_from_file
from deep_stylometry.utils.logger import logging_config

logging_config()
NUM_PROC = psutil.cpu_count(logical=False) - 1


if __name__ == "__main__":
    args = TrainArgparse.parse_known_args()
    config = load_config_from_file(args.config_path)

    dm = train_utils.setup_datamodule(
        config=config,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc if args.num_proc is not None else NUM_PROC,
    )
    model = train_utils.setup_model(config=config)
    trainer = train_utils.setup_trainer(
        config=config,
        logs_dir=args.logs_dir,
        use_wandb=config["use_wandb"],
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.fit(model=model, datamodule=dm)
    if config["do_test"]:
        trainer.test(model=model, datamodule=dm)
