# deep_stylometry/modules/modeling_style_embedding.py

import logging
from typing import Dict, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (AUROC, Accuracy, F1Score, Precision,
                                         Recall)
