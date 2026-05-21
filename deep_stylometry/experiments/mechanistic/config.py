# deep_stylometry/experiments/mechanistic/config.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ProbeSetConfig:
    n_per_tier: int = 50
    target_length: int = 130
    pair_tolerance: int = 10
    global_tolerance: int = 15
    rotation_pool_size: int = 10
    min_documents_per_set: int = 4


@dataclass
class LisaConfig:
    probe_train_size: int = 10000
    probe_eval_size: int = 2000
    feature_categories: List[str] = field(default_factory=lambda: [
        "function_words",
        "sentence_length",
        "punctuation_density",
        "capitalization",
        "type_token_ratio",
        "word_length",
        "hedging",
        "citations",
        "pos_bigrams",
        "discourse_markers",
        "dependency_depth",
    ])


@dataclass
class ModelEntry:
    config: Optional[str] = None
    checkpoint_pattern: Optional[str] = None
    hf_model: Optional[str] = None


@dataclass
class CheckpointsConfig:
    selected_steps: List[Any] = field(default_factory=lambda: [
        0, 500, 1500, 3000, 5000, 10000, 20000, "final"
    ])


@dataclass
class PatchingConfig:
    n_layers: int = 23


@dataclass
class IOConfig:
    output_root: str = "outputs/mechanistic"
    seed: int = 42


@dataclass
class MechanisticConfig:
    base_data_subset: str = "base-4"
    probe_set: ProbeSetConfig = field(default_factory=ProbeSetConfig)
    lisa: LisaConfig = field(default_factory=LisaConfig)
    models: Dict[str, ModelEntry] = field(default_factory=dict)
    checkpoints: CheckpointsConfig = field(default_factory=CheckpointsConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    io: IOConfig = field(default_factory=IOConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "MechanisticConfig":
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        mech_data = data.get("mechanistic", data)
        return cls._from_dict(mech_data)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "MechanisticConfig":
        cfg = cls()
        cfg.base_data_subset = d.get("base_data_subset", cfg.base_data_subset)

        if "probe_set" in d:
            ps = d["probe_set"]
            cfg.probe_set = ProbeSetConfig(**{
                k: v for k, v in ps.items()
                if hasattr(ProbeSetConfig, k) or k in ProbeSetConfig.__dataclass_fields__
            })

        if "lisa" in d:
            ls = d["lisa"]
            cfg.lisa = LisaConfig(**{
                k: v for k, v in ls.items()
                if k in LisaConfig.__dataclass_fields__
            })

        if "models" in d:
            cfg.models = {}
            for name, entry in d["models"].items():
                cfg.models[name] = ModelEntry(**{
                    k: v for k, v in entry.items()
                    if k in ModelEntry.__dataclass_fields__
                })

        if "checkpoints" in d:
            cfg.checkpoints = CheckpointsConfig(
                selected_steps=d["checkpoints"].get(
                    "selected_steps", cfg.checkpoints.selected_steps
                )
            )

        if "patching" in d:
            cfg.patching = PatchingConfig(
                n_layers=d["patching"].get("n_layers", cfg.patching.n_layers)
            )

        if "io" in d:
            io = d["io"]
            cfg.io = IOConfig(
                output_root=io.get("output_root", cfg.io.output_root),
                seed=io.get("seed", cfg.io.seed),
            )

        return cfg
