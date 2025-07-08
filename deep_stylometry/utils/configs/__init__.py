# deep_stylometry/utils/configs/__init__.py

from deep_stylometry.utils.configs.base_config import BaseConfig
from deep_stylometry.utils.configs.data_config import DataConfig
from deep_stylometry.utils.configs.model_config import ModelConfig
from deep_stylometry.utils.configs.train_config import TrainConfig
from deep_stylometry.utils.configs.tune_config import TuneConfig

__all__ = ["BaseConfig", "DataConfig", "ModelConfig", "TrainConfig", "TuneConfig"]
