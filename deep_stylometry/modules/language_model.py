# deep_stylometry/modules/language_model.py

import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM


class LanguageModel(nn.Module):
    def __init__(self, model_name: str):
        super(LanguageModel, self).__init__()

        config = AutoConfig.from_pretrained(model_name)
        is_decoder_model = getattr(config, "is_decoder", False)

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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        return out.hidden_states[-1], out.logits
