# deep_stylometry/utils/tune_utils.py

import logging
from functools import partial
from typing import Any, Dict, Optional

from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import FailureConfig
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from deep_stylometry.utils.train_utils import train_tune


def build_search_space(config: Dict[str, Any]):
    """Builds the search space for hyperparameter tuning.

    Parameters
    ----------
    config: Dict[str, Any]
        Configuration dictionary containing the hyperparameters and their specifications.

    Returns
    -------
    search_space: Dict[str, Any]
        A dictionary representing the search space for hyperparameter tuning.
    """
    search_space = {}
    type_map = {
        "loguniform": tune.loguniform,
        "uniform": tune.uniform,
        "choice": tune.choice,
        "quniform": tune.quniform,
    }
    for param, spec in config.items():
        if not isinstance(spec, dict):
            continue
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
    ray_storage_path: str,
    logs_dir: str,
    use_wandb: bool = False,
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
):
    """Sets up the Ray Tune tuner for hyperparameter tuning. Uses
    HyperOptSearch and AsyncHyperBandScheduler. No checkpointing is done during
    tuning. The goal is to maximize the validation AUROC score before the time
    budget is reached.

    Parameters
    ----------
    config: Dict[str, Any]
        Configuration dictionary containing the hyperparameters and their specifications.
    ray_storage_path: str
        Directory where Ray will save the logs and experiments results.
    use_wandb: bool
        Whether to use Weights & Biases for logging.
    cache_dir: Optional[str]
        Path to the cache directory.
    num_proc: Optional[int]
        Number of processes to use. Default is the number of CPUs minus one.

    Returns
    -------
    tuner: tune.Tuner
        The Ray Tune Tuner object configured for hyperparameter tuning.
    """
    callbacks = []
    if use_wandb:
        logging.info("Using Weights & Biases for logging")
        callbacks.append(
            WandbLoggerCallback(
                project=config["project_name"],
                name=config["experiment_name"],
                group="tune",
                log_config=True,
                log_checkpoints=False,
            )
        )

    search_space = build_search_space(config)

    trainable_fn = partial(
        train_tune,
        base_config=config,
        cache_dir=cache_dir,
        num_proc=num_proc,
        logs_dir=logs_dir,
    )

    trainable_with_resources = tune.with_resources(
        trainable_fn,
        {
            config.get("device", "cpu"): config.get("num_devices_per_trial", 1),
            "cpu": num_proc,
        },  # type: ignore
    )

    asha_scheduler = AsyncHyperBandScheduler(
        time_attr="completed_epoch",  # This corresponds to PTL epochs reported by TuneReportCallback
        # metric="auroc",
        # mode="max",
        max_t=config.get("max_t", 2),
        grace_period=config.get("grace_period", 1),
    )

    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            metric="auroc",
            mode="max",
            search_alg=HyperOptSearch(),
            scheduler=asha_scheduler,
            num_samples=config["num_samples"],
            max_concurrent_trials=config["max_concurrent_trials"],
            time_budget_s=config["time_budget_s"],
            reuse_actors=False,
        ),
        run_config=tune.RunConfig(
            name=config["experiment_name"],
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=False,
                checkpoint_frequency=0,
            ),
            storage_path=ray_storage_path,
            failure_config=FailureConfig(max_failures=0, fail_fast=True),
            stop={"global_step": config.get("max_t", 1000)},
            callbacks=callbacks,
        ),
        param_space=search_space,
    )
    return tuner
