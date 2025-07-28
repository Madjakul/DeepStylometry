# deep_strylometry/utils/helpers.py

from typing import Any

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
