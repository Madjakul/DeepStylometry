# deep_stylometry/utils/configs/base_config.py

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Literal, Union

import yaml

from deep_stylometry.utils.configs.data_config import DataConfig
from deep_stylometry.utils.configs.model_config import ModelConfig
from deep_stylometry.utils.configs.test_config import TestConfig
from deep_stylometry.utils.configs.train_config import TrainConfig
from deep_stylometry.utils.helpers import DictAccessMixin

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig(DictAccessMixin):
    mode: Literal["train", "test"] = "train"
    project_name: str = "deep-stylometry"

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "BaseConfig":
        """Load configuration from YAML file and override defaults."""
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        return cls.from_dict(yaml_data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary, overriding defaults.

        Parameters
        ----------
        config_dict: Dict[str, Any]
            Dictionary containing configuration parameters.
        """
        # Extract mode first if it exists
        mode = config_dict.get("mode", "train")
        config = cls(mode=mode)

        for section_name, section_data in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section_config = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                    else:
                        logger.warning(
                            f"Unknown config key '{key}' in section '{section_name}'"
                        )
            elif hasattr(config, section_name):
                setattr(config, section_name, section_data)
            else:
                logger.warning(f"Unknown config section '{section_name}'")

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
