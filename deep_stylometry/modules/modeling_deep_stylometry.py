# deep_stylometry/modules/modeling_deep_stylometry.py

# TODO: finish the implementation of the DeepStylometry model

import bitsandbytes as bnb
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_stylometry.modules.language_model import LanguageModel
from deep_stylometry.optimizers import SOAP, SophiaG


class DeepStylometry(L.LightningModule):
    """DeepStylometry model."""

    optim_map = {
        "adamw": torch.optim.AdamW,
        "adam8bit": bnb.optim.Adam8bit,
        "soap": SOAP,
        "sophia": SophiaG,
    }

    def __init__(
        self,
        optim_name: str,
        model_name: str,
        batch_size: int,
        lr: float,
        weight_decay: float,
        dropout: float,
        tau: float,
    ):
        super(DeepStylometry, self).__init__()
        self.optim_name = optim_name.lower()
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lm = LanguageModel(model_name)
        self.fc = nn.Linear(self.lm.hidden_size, self.lm.hidden_size)
        self.dropout = dropout
        self.tau = tau

    def configure_optimizers(self):
        optim = self.optim_map[self.optim_name]
        return optim(self.parameters())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.lm(input_ids, attention_mask)
        if self.training:
            out = F.dropout(out, p=self.dropout)
            out = self.fc(out)
            out = F.tanh(out)
        return out

    def training_step(self, batch, batch_idx):
        # TODO: double forward pass into similarity metric to test
        # find a way to get the CLM loss
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
