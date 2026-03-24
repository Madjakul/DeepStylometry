# deep_stylometry/experiments/optuna_pli_search.py
"""Optuna hyperparameter search for Patch-Level Late Interaction (PLI).

Searches over PLI-specific hyperparameters only (backbone/head frozen from
prior runs).  Runs 30 TPE trials with MedianPruner; objective is validation
accuracy after ``max_steps`` training steps on HALvest-Contrastive.

Usage
-----
python -m deep_stylometry.experiments.optuna_pli_search \\
    --config_path configs/train.yml \\
    --processed_ds_dir /path/to/processed \\
    --n_trials 30 \\
    --max_steps 10000 \\
    --output_path configs/best_pli_params.yml
"""

import argparse
import logging
import os
import tempfile
from typing import Optional

import optuna
import yaml
import torch
import lightning as L

from deep_stylometry.modules.modeling_deep_stylometry import DeepStylometry
from deep_stylometry.utils.configs import BaseConfig
from deep_stylometry.utils.train_utils import setup_datamodule

try:
    from optuna.integration import PyTorchLightningPruningCallback
except ModuleNotFoundError:
    # optuna-integration[pytorch_lightning] not installed; use no-op fallback
    class PyTorchLightningPruningCallback(L.Callback):  # type: ignore[no-redef]
        def __init__(self, trial, monitor):
            super().__init__()


logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def build_trial_cfg(base_cfg: BaseConfig, trial: optuna.Trial) -> BaseConfig:
    """Mutate a copy of base_cfg with sampled hyperparameters."""
    import copy
    cfg = copy.deepcopy(base_cfg)

    cfg.model.patch_lambda = trial.suggest_float("patch_lambda", 1e-3, 1.0, log=True)
    cfg.model.gumbel_tau_init = trial.suggest_float("gumbel_tau_init", 0.5, 2.0)
    cfg.model.gumbel_tau_final = trial.suggest_float("gumbel_tau_final", 0.05, 0.5)
    cfg.model.gumbel_anneal_steps = trial.suggest_int(
        "gumbel_anneal_steps", 1000, 20000, step=1000
    )
    cfg.model.patch_cross_attn_dim = trial.suggest_categorical(
        "patch_cross_attn_dim", [64, 128, 256]
    )
    cfg.model.patch_cross_attn_heads = trial.suggest_categorical(
        "patch_cross_attn_heads", [1, 2, 4]
    )
    cfg.model.patch_compression = trial.suggest_categorical(
        "patch_compression", ["mean", "cross_attention"]
    )

    return cfg


def objective(
    trial: optuna.Trial,
    base_cfg: BaseConfig,
    processed_ds_dir: str,
    num_proc: int,
    max_steps: int,
    tmp_dir: str,
) -> float:
    """Train for ``max_steps`` steps, return val accuracy."""
    cfg = build_trial_cfg(base_cfg, trial)
    cfg.train.max_steps = max_steps
    cfg.train.max_epochs = 1  # Prevent early stopping by epoch
    cfg.train.use_wandb = False
    cfg.train.num_devices = 1
    cfg.train.strategy = "auto"
    cfg.train.gather = False

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val/accuracy")

    ckpt_dir = os.path.join(tmp_dir, f"trial_{trial.number}")
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = L.Trainer(
        max_steps=max_steps,
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[pruning_callback],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        log_every_n_steps=max(1, max_steps // 20),
    )

    datamodule = setup_datamodule(cfg, processed_ds_dir, num_proc)

    model = DeepStylometry(cfg)

    try:
        trainer.fit(model, datamodule=datamodule)
    except optuna.exceptions.TrialPruned:
        raise

    val_acc = trainer.callback_metrics.get("val/accuracy", None)
    if val_acc is None:
        return 0.0
    return float(val_acc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_search(
    config_path: str,
    processed_ds_dir: str,
    n_trials: int,
    max_steps: int,
    output_path: str,
    num_proc: int = 4,
    study_name: str = "pli_search",
    storage: Optional[str] = None,
) -> None:
    base_cfg = BaseConfig.from_yaml(config_path)
    # Ensure PLI is enabled
    base_cfg.model.pooling_method = "pli"
    base_cfg.model.patch_method = "learned"

    tmp_dir = tempfile.mkdtemp(prefix="pli_optuna_")
    logging.info(f"Optuna tmp dir: {tmp_dir}")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=max_steps // 5
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial, base_cfg, processed_ds_dir, num_proc, max_steps, tmp_dir
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    logging.info(f"Best params: {best}")
    logging.info(f"Best val accuracy: {study.best_value:.4f}")

    with open(output_path, "w") as f:
        yaml.dump({"best_params": best, "best_val_accuracy": study.best_value}, f)
    logging.info(f"Saved best params to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--processed_ds_dir", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--output_path", type=str, default="configs/best_pli_params.yml")
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--study_name", type=str, default="pli_search")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///pli.db)")
    args = parser.parse_args()

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
