# deep_stylometry/experiments/mechanistic/tests/test_config.py
"""Tests for MechanisticConfig loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from deep_stylometry.experiments.mechanistic.config import (
    MechanisticConfig,
    ModelEntry,
)


_MINIMAL_YAML = textwrap.dedent("""\
    mechanistic:
      base_data_subset: base-4

      probe_set:
        n_per_tier: 25
        target_length: 100

      lisa:
        probe_train_size: 500
        probe_eval_size: 100

      models:
        mean:
          config: configs/test_mean.yml
          checkpoint_pattern: checkpoints/mean
        layerwise:
          config: configs/test_layerwise.yml
          checkpoint_pattern: checkpoints/layerwise
        e5:
          hf_model: intfloat/multilingual-e5-base

      checkpoints:
        selected_steps: [0, 500, "final"]

      patching:
        n_layers: 5

      io:
        output_root: /tmp/test_mechanistic
        seed: 7
""")


class TestConfigFromYaml:
    def test_loads_without_error(self, tmp_path: Path) -> None:
        yml = tmp_path / "mechanistic.yml"
        yml.write_text(_MINIMAL_YAML)
        cfg = MechanisticConfig.from_yaml(yml)
        assert isinstance(cfg, MechanisticConfig)

    def test_model_entries_populated(self, tmp_path: Path) -> None:
        yml = tmp_path / "mechanistic.yml"
        yml.write_text(_MINIMAL_YAML)
        cfg = MechanisticConfig.from_yaml(yml)
        assert "mean" in cfg.models
        assert "layerwise" in cfg.models
        assert "e5" in cfg.models

    def test_mean_entry_fields(self, tmp_path: Path) -> None:
        yml = tmp_path / "mechanistic.yml"
        yml.write_text(_MINIMAL_YAML)
        cfg = MechanisticConfig.from_yaml(yml)
        mean_entry = cfg.models["mean"]
        assert isinstance(mean_entry, ModelEntry)
        assert mean_entry.config == "configs/test_mean.yml"
        assert mean_entry.checkpoint_pattern == "checkpoints/mean"
        assert mean_entry.hf_model is None

    def test_layerwise_entry_fields(self, tmp_path: Path) -> None:
        yml = tmp_path / "mechanistic.yml"
        yml.write_text(_MINIMAL_YAML)
        cfg = MechanisticConfig.from_yaml(yml)
        lw = cfg.models["layerwise"]
        assert lw.config == "configs/test_layerwise.yml"
        assert lw.checkpoint_pattern == "checkpoints/layerwise"

    def test_e5_entry_hf_model(self, tmp_path: Path) -> None:
        yml = tmp_path / "mechanistic.yml"
        yml.write_text(_MINIMAL_YAML)
        cfg = MechanisticConfig.from_yaml(yml)
        e5 = cfg.models["e5"]
        assert e5.hf_model == "intfloat/multilingual-e5-base"
        assert e5.config is None

    def test_scalar_fields(self, tmp_path: Path) -> None:
        yml = tmp_path / "mechanistic.yml"
        yml.write_text(_MINIMAL_YAML)
        cfg = MechanisticConfig.from_yaml(yml)
        assert cfg.probe_set.n_per_tier == 25
        assert cfg.lisa.probe_train_size == 500
        assert cfg.patching.n_layers == 5
        assert cfg.io.seed == 7

    def test_selected_steps_contain_final(self, tmp_path: Path) -> None:
        yml = tmp_path / "mechanistic.yml"
        yml.write_text(_MINIMAL_YAML)
        cfg = MechanisticConfig.from_yaml(yml)
        assert "final" in cfg.checkpoints.selected_steps
        assert 0 in cfg.checkpoints.selected_steps
