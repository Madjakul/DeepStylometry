# deep_stylometry/utils/configs/base_config.py

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import yaml

from deep_stylometry.utils.configs.data_config import DataConfig
from deep_stylometry.utils.configs.model_config import ModelConfig
from deep_stylometry.utils.configs.train_config import TrainConfig
from deep_stylometry.utils.configs.tune_config import TuneConfig
from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class BaseConfig(DictAccessMixin):
    mode: Literal["train", "tune"] = "train"

    project_name: str = "deep-stylometry"
    group_name: str = "train-deep-stylometry-512-se"
    do_train: bool = True
    do_test: bool = False

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    _train: TrainConfig = TrainConfig()
    _tune: TuneConfig = TuneConfig()

    @property
    def train(self) -> TrainConfig:
        """Access the training configuration."""
        if self.mode != "train":
            raise ValueError(
                "Cannot access training configuration when mode is not 'train'."
            )
        return self._train

    @property
    def tune(self) -> TuneConfig:
        """Access the tuning configuration."""
        if self.mode != "tune":
            raise ValueError(
                "Cannot access tuning configuration when mode is not 'tune'."
            )
        return self._tune

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "BaseConfig":
        """Load configuration from YAML file and override defaults."""
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        return cls.from_dict(yaml_data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary, overriding defaults."""
        config = cls()

        for section_name, section_data in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section_config = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                    else:
                        logging.warning(
                            f"Unknown config key '{key}' in section '{section_name}'"
                        )
            elif hasattr(config, section_name):
                setattr(config, section_name, section_data)
            else:
                logging.warning(f"Unknown config section '{section_name}'")

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field in fields(self):
            section_config = getattr(self, field.name)
            if hasattr(section_config, "__dict__"):
                result[field.name] = {
                    f.name: getattr(section_config, f.name)
                    for f in fields(section_config)
                }
            else:
                result[field.name] = section_config
        return result

    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save current configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
