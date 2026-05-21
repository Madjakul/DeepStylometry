# deep_stylometry/modules/language_model.py

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from transformers import AutoConfig, AutoModel

from deep_stylometry.utils.helpers import resolve_lightning_precision

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class LanguageModel(nn.Module):
    """Thin wrapper around a HuggingFace AutoModel encoder.

    Loads the model and config from ``cfg.model.base_checkpoint``, resolves
    the precision dtype, and returns per-token hidden states from the final
    layer.

    Parameters
    ----------
    cfg : BaseConfig
        Global configuration. ``cfg.model.base_checkpoint`` and
        ``cfg.model.attn_implementation`` are used at construction time.
    """

    def __init__(self, cfg: "BaseConfig") -> None:
        super(LanguageModel, self).__init__()
        self.cfg = cfg
        _, torch_dtype = resolve_lightning_precision(cfg.train.precision)

        config = AutoConfig.from_pretrained(
            self.cfg.model.base_checkpoint, torch_dtype=torch_dtype
        )

        self.model = AutoModel.from_pretrained(
            cfg.model.base_checkpoint,
            config=config,
            attn_implementation=cfg.model.attn_implementation,
        )

        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        attention_mask: Int[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch seq hidden"]:
        """Run the encoder and return last-layer hidden states.

        Parameters
        ----------
        input_ids : Int[Tensor, "batch seq"]
            Token IDs.
        attention_mask : Int[Tensor, "batch seq"]
            Binary attention mask.

        Returns
        -------
        Float[Tensor, "batch seq hidden"]
            Per-token hidden states from the final transformer layer.
        """

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        return out.last_hidden_state
