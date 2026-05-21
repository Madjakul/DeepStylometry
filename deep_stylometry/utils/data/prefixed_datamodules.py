# deep_stylometry/utils/data/prefixed_datamodules.py
"""Prefixed datamodule variants for zero-shot evaluation.

Wraps ``PAN19Datamodule`` and ``HALvestContrastiveDatamodule`` with a
configurable ``query_prefix`` / ``passage_prefix`` prepended to each text
column before tokenisation. Used for models trained with asymmetric
prefixes (E5: ``query: `` / ``passage: ``; BGE: ``Represent this sentence
for retrieval: `` on the query side; etc.).
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import lightning as L

from deep_stylometry.utils.data.halvest_datamodule import HALvestContrastiveDatamodule
from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
from deep_stylometry.utils import train_utils

if TYPE_CHECKING:
    from deep_stylometry.utils.configs.base_config import BaseConfig


class PrefixedPAN19Datamodule(PAN19Datamodule):
    """PAN19 datamodule that prepends prefixes to query / positive / negative."""

    def __init__(
        self,
        *args,
        query_prefix: str = "",
        passage_prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def tokenize(
        self, batch: Dict[str, List[Any]], indices: List[int]
    ) -> Dict[str, Any]:
        batch = dict(batch)  # shallow copy; preserves non-text columns if any
        batch["query"] = [self.query_prefix + t for t in batch["query"]]
        batch["positive"] = [self.passage_prefix + t for t in batch["positive"]]
        batch["negative"] = [self.passage_prefix + t for t in batch["negative"]]
        return super().tokenize(batch, indices)


class PrefixedHALvestDatamodule(HALvestContrastiveDatamodule):
    """HALvest datamodule that prepends prefixes to query / positive / negative."""

    def __init__(
        self,
        *args,
        query_prefix: str = "",
        passage_prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def tokenize(
        self, batch: Dict[str, List[Any]], indices: List[int]
    ) -> Dict[str, Any]:
        batch = dict(batch)
        batch["query"] = [self.query_prefix + t for t in batch["query"]]
        batch["positive"] = [self.passage_prefix + t for t in batch["positive"]]
        batch["negative"] = [self.passage_prefix + t for t in batch["negative"]]
        return super().tokenize(batch, indices)


def setup_zero_shot_datamodule(
    cfg: "BaseConfig",
    processed_ds_dir: str,
    num_proc: int,
    cache_dir: Optional[str] = None,
    query_prefix: str = "",
    passage_prefix: str = "",
) -> L.LightningDataModule:
    """Return a prefixed datamodule when prefixes are set, else the standard one.

    When both prefixes are empty, delegates to ``train_utils.setup_datamodule``
    so the behaviour is identical to a non-prefixed zero-shot run.
    """
    if not (query_prefix or passage_prefix):
        return train_utils.setup_datamodule(cfg, processed_ds_dir, num_proc, cache_dir)

    kwargs: dict = dict(
        cfg=cfg,
        processed_ds_dir=processed_ds_dir,
        num_proc=num_proc,
        cache_dir=cache_dir,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
    if cfg.data.ds_name == "pan19":
        kwargs["pan19_zip"] = os.environ.get("PAN19_ZIP")
        return PrefixedPAN19Datamodule(**kwargs)
    if cfg.data.ds_name == "halvest":
        return PrefixedHALvestDatamodule(**kwargs)
    raise ValueError(
        f"Zero-shot prefixes not supported for ds_name='{cfg.data.ds_name}'. "
        f"Expected 'pan19' or 'halvest'."
    )
