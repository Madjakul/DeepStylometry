# deep_stylometry/utils/tune_utils.py

import logging
from functools import partial
from typing import Any, Dict, Optional

from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import FailureConfig
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from deep_stylometry.utils.configs import BaseConfig
from deep_stylometry.utils.train_utils import train_tune


def make_tuners(o) -> Any:
    if isinstance(o, dict) and "type" in o:
        t = o["type"]
        if t == "loguniform":
            return tune.loguniform(o["min"], o["max"])
        if t == "uniform":
            return tune.uniform(o["min"], o["max"])
        if t == "quniform":
            return tune.quniform(o["min"], o["max"], o["q"])
        if t == "choice":
            return tune.choice(o["values"])
        raise ValueError(f"Unknown tune type {t!r}")
    elif isinstance(o, dict):
        return {k: make_tuners(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [make_tuners(v) for v in o]
    else:
        return o


def setup_tuner(
    config: BaseConfig,
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
    config: BaseConfig
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
                project=config.project_name,
                name=f"""tune-{config.model.base_model_name}-{config.data.ds_name}
                -pooling:{config.model.pooling_method}-softmax:{config.model.use_softmax}
                -dist:{config.model.distance_weightning}""",
                group=config.group_name,
                log_config=True,
                log_checkpoints=False,
            )
        )

    param_space = make_tuners(config)

    trainable_fn = partial(
        train_tune,
        cache_dir=cache_dir,
        num_proc=num_proc,
        logs_dir=logs_dir,
    )

    trainable_with_resources = tune.with_resources(
        trainable_fn,
        {
            config.tune.device: config.tune.num_devices_per_trial,
            "cpu": num_proc,
        },  # type: ignore
    )

    asha_scheduler = AsyncHyperBandScheduler(
        time_attr="completed_epoch",
        max_t=config.tune.max_t,
        grace_period=config.tune.grace_period,
    )

    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            metric="auroc",
            mode="max",
            search_alg=HyperOptSearch(),
            scheduler=asha_scheduler,
            num_samples=config.tune.num_samples,
            max_concurrent_trials=config.tune.max_concurrent_trials,
            time_budget_s=config.tune.time_budget_s,
            reuse_actors=False,
        ),
        run_config=tune.RunConfig(
            name=config.group_name,
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=False,
                checkpoint_frequency=0,
            ),
            storage_path=ray_storage_path,
            failure_config=FailureConfig(max_failures=0, fail_fast=True),
            stop={"completed_epoch": config.tune.max_t},
            callbacks=callbacks,
        ),
        param_space=param_space,
    )
    return tuner
