# deep_stylometry/utils/tune_utils.py

from functools import partial
from typing import Any, Optional

from ray.tune import FailureConfig
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from deep_stylometry.utils.configs.base_config import BaseConfig
from deep_stylometry.utils.train_utils import train_tune
from ray import tune


def make_param_space(o: Any) -> Optional[Any]:
    """Build a nested param_space that mirrors `o`:

    Parameters
    ----------
    o : Any
        The object to convert into a Ray Tune parameter space.

    Attributes
    ----------
    Any
        - If `o` is a dict with "type", return the corresponding tune.* sampler.
        - If `o` is a dict without "type", recurse into items and keep only
          non-None children; return that dict (or None if empty).
        - If `o` is a list, recurse on each element; if any element yields a
          sampler, return the list of samplers (or None if none).
        - If `o` is a scalar (str/int/float/bool/None), wrap in tune.choice([o]).
    """
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

    # Nested dict -> recurse
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            child = make_param_space(v)
            if child is not None:
                out[k] = child
        return out or None

    # List -> recurse
    if isinstance(o, list):
        recursed = [make_param_space(v) for v in o]
        recursed = [v for v in recursed if v is not None]
        return recursed or None

    # Scalar constant -> wrap as a single‑choice sampler
    return tune.choice([o])


def setup_tuner(
    config: BaseConfig,
    ray_storage_path: str,
    cache_dir: Optional[str] = None,
) -> tune.Tuner:
    """Set up the Ray Tune tuner for hyper-parameter tuning.

    Parameters
    ----------
    config: BaseConfig
        Configuration object containing the tuning parameters.
    ray_storage_path: str
        Directory where Ray will save the logs and experiment results.
    cache_dir: Optional[str]
        Directory to cache the dataset. If None, defaults to the current working
        directory.

    Returns
    -------
    tuner: tune.Tuner
        The configured Ray Tune tuner ready for hyper-parameter tuning.
    """
    callbacks = []
    param_space = make_param_space(config.to_dict())

    trainable_fn = partial(
        train_tune,
        cache_dir=cache_dir,
    )

    trainable_with_resources = tune.with_resources(
        trainable_fn,
        {
            config.tune.device: config.tune.num_devices_per_trial,
            "cpu": config.tune.num_cpus_per_trial,
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
            metric=config.tune.metric,
            mode=config.tune.mode,
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
            verbose=2,
        ),
        param_space=param_space,
    )
    return tuner
