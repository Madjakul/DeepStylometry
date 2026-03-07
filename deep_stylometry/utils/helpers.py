# deep_strylometry/utils/helpers.py

import logging
import os
import random
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

WIDTH = 88


class DictAccessMixin:
    """Mixin to add dictionary-like access to dataclass instances."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def get_tokenizer(model_name: str, **kwargs) -> "PreTrainedTokenizerBase":
    """Get a tokenizer from the model name.

    Parameters
    ----------
    model_name: str
        Name of the model.

    Returns
    -------
    tokenizer: transformers.PretrainedTokenizerBase
        Tokenizer for the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("Tokenizer has neither pad_token nor eos_token defined.")
    return tokenizer


def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


def resolve_lightning_precision(
    requested_precision: str,
) -> tuple[str, torch.dtype]:
    if requested_precision == "bf16-mixed":
        bf16_ok = (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and torch.cuda.get_device_capability(0)[0] >= 8
        )
        if bf16_ok:
            logging.info("Using bfloat16 mixed precision.")
            return "bf16-mixed", torch.bfloat16
        else:
            logging.warning(
                "Bfloat16 mixed precision is not supported on this hardware. Falling back to float16 mixed precision."
            )
            return "16-mixed", torch.float16

    if requested_precision == "16-mixed":
        logging.info("Using float16 mixed precision.")
        return "16-mixed", torch.float16

    if requested_precision in ("32", "32-true"):
        logging.info("Using float32 precision.")
        return "32-true", torch.float32

    raise ValueError(f"Unknown Lightning precision: {requested_precision}")
