# deep_stylometry/modules/language_model.py

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from transformers import AutoConfig, AutoModel

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class LanguageModel(nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super(LanguageModel, self).__init__()
        self.cfg = cfg

        config = AutoConfig.from_pretrained(self.cfg.model.base_checkpoint)
        self.model = AutoModel.from_pretrained(cfg.model.base_checkpoint, config=config)

        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        attention_mask: Int[torch.Tensor, "batch seq"],
    ) -> Tuple[Float[torch.Tensor, ""], Float[torch.Tensor, "batch seq hidden"]]:
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1]
