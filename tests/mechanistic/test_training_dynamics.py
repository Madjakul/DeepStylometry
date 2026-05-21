# deep_stylometry/experiments/mechanistic/tests/test_training_dynamics.py
"""Tests for training_dynamics — step-column serialization."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest


class TestStepColumnSerialization:
    def test_mixed_int_str_step_writes_parquet(self, tmp_path: Path) -> None:
        """Mixed int/str step values must survive a round-trip through Parquet."""
        summary_rows: List[Dict[str, Any]] = []
        for step in [0, 500, 1500, "final"]:
            summary_rows.append({
                "model_id": "mean",
                "step": str(step),
                "inflection_layer": 5,
                "emergence_layer": 3,
                "mean_recovery_at_inflection": 42.0,
                "n_tier_a": 50,
            })

        df = pd.DataFrame(summary_rows)
        out = tmp_path / "summary.parquet"
        # Must not raise ArrowInvalid
        df.to_parquet(out, index=False)
        assert out.exists()

        df_back = pd.read_parquet(out)
        assert set(df_back["step"].tolist()) == {"0", "500", "1500", "final"}
        # pandas 2.0+ may return StringDtype instead of object; both are string-like
        assert df_back["step"].dtype == object or hasattr(df_back["step"].dtype, "na_value")

    def test_step_column_all_strings(self, tmp_path: Path) -> None:
        """str(step) normalisation makes all step values uniform strings."""
        steps = [0, 500, 1500, 3000, 5000, 10000, 20000, "final"]
        rows = [{"step": str(s)} for s in steps]
        df = pd.DataFrame(rows)
        out = tmp_path / "step_check.parquet"
        df.to_parquet(out, index=False)
        df_back = pd.read_parquet(out)
        for val in df_back["step"]:
            assert isinstance(val, str), f"Expected str, got {type(val)}: {val!r}"
