# deep_stylometry/modules/modeling_style_embedding.py

import logging
from typing import Dict, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall
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

        self.val_auroc = AUROC(
            task="multiclass", num_classes=batch_size, average="macro"
        )
        self.val_f1 = F1Score()
        self.val_precision = Precision()
        self.val_recall = Recall()
        self.test_auroc = AUROC(
            task="multiclass", num_classes=batch_size, average="macro"
        )
        self.test_f1 = F1Score()
        self.test_precision = Precision()
        self.test_recall = Recall()

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
