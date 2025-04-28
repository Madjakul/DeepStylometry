# deep_stylometry/modules/language_model.py

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM


class LanguageModel(nn.Module):
    def __init__(self, model_name: str, is_decoder_model: bool):
        super(LanguageModel, self).__init__()

        config = AutoConfig.from_pretrained(model_name)

        if is_decoder_model:
            logging.info(f"Loading pretrained decoder from {model_name}.")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
            self.is_decoder = True
        else:
            logging.info(f"Loading pretrained encoder from {model_name}.")
            self.model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
            self.is_decoder = False

        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        # out = self.model(
        #     input_ids, attention_mask=attention_mask, output_hidden_states=True
        # )
        # return out.hidden_states[-1], out.logits
        if self.is_decoder and labels is None:
            labels = input_ids.clone()
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = out.loss if labels is not None else 0.0
        last_hidden_states = out.hidden_states[-1]
        return loss, last_hidden_states
