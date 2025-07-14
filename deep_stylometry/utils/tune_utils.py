# deep_stylometry/utils/tune_utils.py

import logging
from functools import partial
from typing import Any, Optional

from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import FailureConfig
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from deep_stylometry.utils.configs.base_config import BaseConfig
from deep_stylometry.utils.train_utils import train_tune
from ray import tune


def make_param_space(o: Any) -> Optional[Any]:
    """Build a nested param_space that mirrors `o`:

    • If `o` is a dict with "type", return the corresponding tune.* sampler.
    • If `o` is a dict without "type", recurse into items and keep only
      non-None children; return that dict (or None if empty).
    • If `o` is a list, recurse on each element; if any element yields a
      sampler, return the list of samplers (or None if none).
    • If `o` is a scalar (str/int/float/bool/None), wrap in tune.choice([o]).
    """
    # 1) Actual sampler directive?
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

    # 2) Nested dict -> recurse
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            child = make_param_space(v)
            if child is not None:
                out[k] = child
        return out or None

    # 3) List -> recurse
    if isinstance(o, list):
        recursed = [make_param_space(v) for v in o]
        recursed = [v for v in recursed if v is not None]
        return recursed or None

    # 4) Scalar constant -> wrap as a single‑choice sampler
    #    (so Ray still hands it back to you in the same nested shape)
    return tune.choice([o])


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

    param_space = make_param_space(config.to_dict())

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
        },
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
