# train.py

import logging
import os

from deep_stylometry.modules import DeepStylometry
from deep_stylometry.utils import train_utils
from deep_stylometry.utils.argparsers import TrainArgparse
from deep_stylometry.utils.configs import BaseConfig
from deep_stylometry.utils.helpers import set_seed
from deep_stylometry.utils.logger import logging_config

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

set_seed()
logging_config()


if __name__ == "__main__":
    args = TrainArgparse.parse_known_args()
    cfg = BaseConfig(mode="train").from_yaml(args.config_path)

    logger.info("Preparing data module...")
    dm = train_utils.setup_datamodule(
        cfg=cfg,
        processed_ds_dir=args.processed_ds_dir,
        num_proc=args.num_proc,
        cache_dir=args.cache_dir,
    )

    logger.info("=== Fine-tuning ===")
    model = DeepStylometry(cfg)

    trainer = train_utils.setup_trainer(
        cfg=cfg,
        model=model,
        logs_dir=args.logs_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.fit(model=model, datamodule=dm)
    logger.info("=== Fine-tuning finished ===")
