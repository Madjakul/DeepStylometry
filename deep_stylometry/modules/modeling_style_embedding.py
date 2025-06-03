# deep_stylometry/modules/modeling_style_embedding.py

import logging
from typing import Dict, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import BinaryAUROC, HitRate, ReciprocalRank
from transformers import AutoModel, get_linear_schedule_with_warmup


class StyleEmbedding(L.LightningModule):

    def __init__(
        self,
        base_model_name: str = "AnnaWegmann/Style-Embedding",
        batch_size: int = 8,
        lr: float = 2e-05,
        weight_decay=1e-2,
    ):
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        # Test metrics
        self.test_auroc = BinaryAUROC()
        self.test_hr1 = HitRate(k=1)
        self.test_hr5 = HitRate(k=5)
        self.test_hr10 = HitRate(k=10)
        self.test_rr = ReciprocalRank()

        self.model = AutoModel.from_pretrained(base_model_name)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
