# deep_stylometry/modules/language_model.py

import torch
import torch.nn as nn
import transformers


class LanguageModel(nn.Module):
    def __init__(self, model_name: str):
        super(LanguageModel, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_ids, attention_mask)
        return out
