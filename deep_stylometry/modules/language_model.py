# deep_stylometry/modules/language_model.py

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class LanguageModel(nn.Module):

    def __init__(self, cfg: "BaseConfig"):
        super(LanguageModel, self).__init__()
        self.cfg = cfg

        config = AutoConfig.from_pretrained(self.cfg.model.base_model_name)

        if self.cfg.model.is_decoder_model:
            logging.info(
                f"Loading pretrained decoder from {self.cfg.model.base_model_name}."
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.base_model_name, config=config
            )
        else:
            logging.info(
                f"Loading pretrained encoder from {self.cfg.model.base_model_name}."
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.cfg.model.base_model_name, config=config
            )

        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        attention_mask: Int[torch.Tensor, "batch seq"],
        labels: Optional[Int[torch.Tensor, "batch seq"]] = None,
    ) -> Tuple[Float[torch.Tensor, ""], Float[torch.Tensor, "batch seq hidden"]]:
        if self.cfg.model.is_decoder_model and labels is None:
            labels = input_ids.clone()
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = out.loss if out.loss is not None else torch.tensor(0.0)
        last_hidden_states = out.hidden_states[-1]
        return loss, last_hidden_states
