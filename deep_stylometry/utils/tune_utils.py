# deep_stylometry/utils/tune_utils.py

import os
from functools import partial
from typing import Any, Dict, Optional

import lightning as L
import ray
from ray import tune
from ray.air import FailureConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from deep_stylometry.utils.train_utils import train_tune


def build_search_space(config: Dict[str, Any]):
    search_space = {}
    type_map = {
        "loguniform": tune.loguniform,
        "uniform": tune.uniform,
        "choice": tune.choice,
        "quniform": tune.quniform,
    }
    for param, spec in config["tunable"].items():
        if spec["type"] not in ("choice", "quniform"):
            search_space[param] = type_map[spec["type"]](
                spec["min"],
                spec["max"],
            )
        elif spec["type"] == "quniform":
            search_space[param] = type_map[spec["type"]](
                spec["min"],
                spec["max"],
                spec["q"],
            )
        elif spec["type"] == "choice":
            search_space[param] = type_map[spec["type"]](spec["values"])
    return search_space


def setup_tuner(
    config: Dict[str, Any],
    device: str,
    ray_storage_path: str,
    use_wandb: bool = False,
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
):
    callbacks = []
    callbacks.append(
        TuneReportCheckpointCallback(
            {
                "loss": "val_total_loss",
                "auroc": "val_auroc",
                "f1": "val_f1",
                "precision": "val_precision",
                "recall": "val_recall",
            },
            on="validation_end",
        )
    )
    if use_wandb:
        callbacks.append(
            WandbLoggerCallback(
                project=config["project_name"],
                name=config["experiment_name"],
                group="tuning",
                log_config=True,
                log_checkpoints=False,
            )
        )

    search_space = build_search_space(config)

    tuner = tune.Tuner(
        partial(
            train_tune,
            base_config=config,
            device=device,
            cache_dir=cache_dir,
            num_proc=num_proc,
        ),
        tune_config=tune.TuneConfig(
            metric="auroc",
            mode="max",
            search_alg=HyperOptSearch(),
            scheduler=AsyncHyperBandScheduler(),
            num_samples=config["num_samples"],
            max_concurrent_trials=config["max_concurrent_trials"],
            time_budget_s=config["time_budget_s"],
        ),
        run_config=tune.RunConfig(
            name=config["experiment_name"],
            storage_path=ray_storage_path,
            failure_config=FailureConfig(max_failures=config["max_failures"]),
            stop={"training_iteration": config["max_epochs"]},
            callbacks=callbacks,
        ),
        param_space=search_space,
    )
    return tuner
