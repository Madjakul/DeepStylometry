# deep_stylometry/modules/language_model.py

import torch
import torch.nn as nn
import transformers


class LanguageModel(nn.Module):
    def __init__(self, model_name: str):
        super(LanguageModel, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        self.is_decoder = (
            hasattr(self.model.config, "is_decoder") and self.model.config.is_decoder
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_ids, attention_mask, output_hidden_states=True)
        return out.last_hidden_state
