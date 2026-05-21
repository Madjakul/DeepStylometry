# deep_stylometry/experiments/mechanistic/tests/test_io_utils.py
"""Tests for IO utilities — deterministic path generation."""

import tempfile
from pathlib import Path

import pytest

from deep_stylometry.experiments.mechanistic.io_utils import (
    _step_label,
    phase0_path,
    phase1_probe_path,
    phase2_path,
    phase3_path,
    phase4_path,
    activation_cache_path,
    output_exists,
)


class TestStepLabel:
    def test_final(self):
        assert _step_label("final") == "final"
        assert _step_label(0) == "step_0000"
        assert _step_label(500) == "step_0500"
        assert _step_label(5000) == "step_5000"
        assert _step_label(20000) == "step_20000"

    def test_zero_padded(self):
        assert _step_label(1) == "step_0001"
        assert _step_label(9999) == "step_9999"
        assert _step_label(10000) == "step_10000"


class TestPathStability:
    """Paths must be deterministic across calls."""

    def test_phase0_stable(self, tmp_path):
        p1 = phase0_path(tmp_path, "probe_set.json")
        p2 = phase0_path(tmp_path, "probe_set.json")
        assert p1 == p2

    def test_phase1_probe_stable(self, tmp_path):
        p1 = phase1_probe_path(tmp_path, "mean", "final", "probes.npz")
        p2 = phase1_probe_path(tmp_path, "mean", "final", "probes.npz")
        assert p1 == p2

    def test_phase2_stable(self, tmp_path):
        p1 = phase2_path(tmp_path, "li", 5000, "recovery.npz")
        p2 = phase2_path(tmp_path, "li", 5000, "recovery.npz")
        assert p1 == p2

    def test_different_models_different_paths(self, tmp_path):
        p_mean = phase2_path(tmp_path, "mean", "final", "recovery.npz")
        p_li = phase2_path(tmp_path, "li", "final", "recovery.npz")
        assert p_mean != p_li

    def test_different_steps_different_paths(self, tmp_path):
        p1 = phase2_path(tmp_path, "mean", 5000, "recovery.npz")
        p2 = phase2_path(tmp_path, "mean", 10000, "recovery.npz")
        assert p1 != p2


class TestOutputExists:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text("{}")
        assert output_exists(f)

    def test_missing_file(self, tmp_path):
        f = tmp_path / "missing.json"
        assert not output_exists(f)

    def test_all_must_exist(self, tmp_path):
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        f1.write_text("{}")
        assert not output_exists(f1, f2)
        f2.write_text("{}")
        assert output_exists(f1, f2)


class TestActivationCachePath:
    def test_path_structure(self, tmp_path):
        p = activation_cache_path(tmp_path, "mean", "final", "probe_train")
        assert "mean" in str(p)
        assert "final" in str(p)
        assert "probe_train" in str(p)
        assert p.suffix == ".npz"

    def test_step_in_path(self, tmp_path):
        p = activation_cache_path(tmp_path, "li", 5000, "probe_train")
        assert "step_5000" in str(p)
