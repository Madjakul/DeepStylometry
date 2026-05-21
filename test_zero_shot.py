# test_zero_shot.py
"""Zero-shot evaluation of a pretrained encoder on PAN19 or HALvest.

Loads the base encoder from ``cfg.model.base_checkpoint`` via HuggingFace and
runs the standard ``TestEvalCallback`` pipeline with mean pooling. No
fine-tuned checkpoint is loaded — this is a control baseline showing how
much of the authorship-attribution structure a general-purpose retrieval
model recovers before any stylometric fine-tuning.

Prefixes are read from the environment:
  ZERO_SHOT_QUERY_PREFIX   — prepended to the query text (E5: "query: ")
  ZERO_SHOT_PASSAGE_PREFIX — prepended to positive/negative  (E5: "passage: ")

When both are empty the behaviour is identical to a vanilla zero-shot run.
"""

import logging
import os
import sys

import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from deep_stylometry.callbacks import TestEvalCallback
from deep_stylometry.modules import DeepStylometry
from deep_stylometry.utils.argparsers import TestArgparse
from deep_stylometry.utils.configs import BaseConfig
from deep_stylometry.utils.data.prefixed_datamodules import setup_zero_shot_datamodule
from deep_stylometry.utils.helpers import resolve_lightning_precision, set_seed
from deep_stylometry.utils.logger import logging_config

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

set_seed()
logging_config()

if __name__ == "__main__":
    # TestArgparse marks --checkpoint_path as required; inject a sentinel so the
    # parser does not error when the flag is legitimately absent for zero-shot runs.
    if "--checkpoint_path" not in sys.argv:
        sys.argv.extend(["--checkpoint_path", "unused"])

    args = TestArgparse.parse_known_args()
    cfg = BaseConfig(mode="test").from_yaml(args.config_path)

    if args.test_subset is not None:
        cfg.data.test_subset = args.test_subset
    if args.ds_name is not None:
        cfg.data.ds_name = args.ds_name

    assert cfg.model.pooling_method == "mean", (
        "Zero-shot entry point only supports pooling_method='mean'."
    )

    query_prefix = os.environ.get("ZERO_SHOT_QUERY_PREFIX", "")
    passage_prefix = os.environ.get("ZERO_SHOT_PASSAGE_PREFIX", "")
    if query_prefix or passage_prefix:
        logging.info(
            f"Using prefixes: query={query_prefix!r} passage={passage_prefix!r}"
        )

    logging.info("Preparing data module...")
    dm = setup_zero_shot_datamodule(
        cfg=cfg,
        processed_ds_dir=args.processed_ds_dir,
        num_proc=args.num_proc,
        cache_dir=args.cache_dir,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )

    prefix_tag = "prefixed" if (query_prefix or passage_prefix) else "raw"
    name = (
        f"zeroshot__{cfg.model.base_checkpoint}__{cfg.data.ds_name}"
        f"__pooling-mean__{cfg.data.test_subset}__{prefix_tag}"
    ).replace("/", "-").lower()

    loggers = []
    if cfg.test.use_wandb:
        loggers.append(WandbLogger(project=cfg.project_name, name=name))
    loggers.append(CSVLogger(save_dir=args.logs_dir, name=name))

    logging.info(
        f"Instantiating fresh model from {cfg.model.base_checkpoint} (no fine-tuning)."
    )
    model = DeepStylometry(cfg)
    precision, _ = resolve_lightning_precision(cfg.test.precision)

    trainer = L.Trainer(
        accelerator=cfg.test.device,
        devices=cfg.test.num_devices,
        logger=loggers,
        callbacks=[TestEvalCallback(cfg)],
        precision=precision,
    )

    logging.info("=== Starting Zero-Shot Evaluation ===")
    trainer.test(model=model, datamodule=dm)
    logging.info("=== Zero-Shot Evaluation Finished ===")
