# deep_stylometry/experiments/mechanistic/io_utils.py
"""Deterministic output path factory and resume helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def _step_label(step: Union[int, str]) -> str:
    """Normalise a checkpoint step to a directory name."""
    if step == "final" or str(step) == "final":
        return "final"
    return f"step_{int(step):04d}"


def output_root(cfg_or_str: Union[str, Path, "MechanisticConfig"]) -> Path:  # type: ignore[name-defined]  # noqa: F821
    if hasattr(cfg_or_str, "io"):
        root = cfg_or_str.io.output_root
    else:
        root = str(cfg_or_str)
    return Path(root)


def phase0_path(root: Union[str, Path], filename: str) -> Path:
    p = Path(root) / "phase0_probe_set" / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def phase1_lisa_path(root: Union[str, Path], filename: str) -> Path:
    p = Path(root) / "phase1_lisa" / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def phase1_probe_path(
    root: Union[str, Path],
    model_id: str,
    step: Union[int, str],
    filename: str,
) -> Path:
    p = Path(root) / "phase1_probes" / model_id / _step_label(step) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def phase2_path(
    root: Union[str, Path],
    model_id: str,
    step: Union[int, str],
    filename: str,
) -> Path:
    p = Path(root) / "phase2_patching" / model_id / _step_label(step) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def phase3_path(root: Union[str, Path], filename: str) -> Path:
    p = Path(root) / "phase3_dynamics" / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def phase4_path(
    root: Union[str, Path],
    model_id: str,
    filename: str,
) -> Path:
    p = Path(root) / "phase4_distractors" / model_id / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def figures_path(root: Union[str, Path], filename: str) -> Path:
    p = Path(root) / "figures" / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def activation_cache_path(
    root: Union[str, Path],
    model_id: str,
    step: Union[int, str],
    split: str,
) -> Path:
    """Path for cached hidden-state arrays (large intermediate files)."""
    p = Path(root) / "activation_cache" / model_id / _step_label(step) / f"{split}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def output_exists(*paths: Path) -> bool:
    return all(p.exists() for p in paths)
