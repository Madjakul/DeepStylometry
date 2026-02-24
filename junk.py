import psutil
from deep_stylometry.utils.configs import BaseConfig
from deep_stylometry.utils.data import HALvestContrastiveDatamodule

num_proc = psutil.cpu_count(logical=False)
cfg = BaseConfig.from_yaml("./configs/train.yml")
dm = HALvestContrastiveDatamodule(
    cfg, "../Datasets/deep-stylometry/answerdotai-modernbert-base/no-padding", num_proc
)
dm.setup(stage="test")
