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
        self.model = AutoModel.from_pretrained(base_model_name)
