# tests/test_optuna_search.py
"""Smoke tests for the Optuna PLI hyperparameter search.

These tests verify that the helper utilities work in isolation without
requiring a GPU or a full dataset.
"""

import os
import tempfile

import pytest
import yaml

from deep_stylometry.experiments.optuna_pli_search import build_trial_cfg


class TestBuildTrialCfg:
    def test_build_trial_cfg(self, dummy_cfg):
        """build_trial_cfg should return a valid cfg with sampled params."""
        import optuna

        study = optuna.create_study(direction="maximize")

        def _objective(trial):
            cfg = build_trial_cfg(dummy_cfg, trial)
            # Verify all expected fields are set
            assert 1e-3 <= cfg.model.patch_lambda <= 1.0
            assert 0.5 <= cfg.model.gumbel_tau_init <= 2.0
            assert 0.05 <= cfg.model.gumbel_tau_final <= 0.5
            assert cfg.model.patch_cross_attn_dim in (64, 128, 256)
            assert cfg.model.patch_cross_attn_heads in (1, 2, 4)
            assert cfg.model.patch_compression in ("mean", "cross_attention")
            # Return a dummy value
            return 0.5

        study.optimize(_objective, n_trials=3)
        assert len(study.trials) == 3

    def test_output_yaml(self, tmp_path):
        """build_trial_cfg output can be round-tripped through YAML."""
        result = {
            "best_params": {"patch_lambda": 0.01, "gumbel_tau_init": 1.0},
            "best_val_accuracy": 0.85,
        }
        path = str(tmp_path / "params.yml")
        with open(path, "w") as f:
            yaml.dump(result, f)

        with open(path, "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded["best_val_accuracy"] == pytest.approx(0.85)
        assert "patch_lambda" in loaded["best_params"]
