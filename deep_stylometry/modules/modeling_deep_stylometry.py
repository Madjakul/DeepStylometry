# deep_stylometry/modules/modeling_deep_stylometry.py

from typing import Dict, Optional

import bitsandbytes as bnb
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_stylometry.modules.clm_loss import CLMLoss
from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.modules.language_model import LanguageModel
from deep_stylometry.optimizers import SOAP, SophiaG


class DeepStylometry(L.LightningModule):

    optim_map = {
        "adamw": torch.optim.AdamW,
        "adam8bit": bnb.optim.Adam8bit,
        "soap": SOAP,
        "sophia": SophiaG,
    }

    def __init__(
        self,
        optim_name: str,
        base_model_name: str,
        batch_size: int,
        seq_len: int,
        lr: float = 2e-5,
        dropout: float = 0.1,
        weight_decay: float = 1e-2,
        clm_weight: float = 1.0,
        do_late_interaction: bool = False,
        do_distance: bool = False,
        exp_decay: bool = False,
        alpha: float = 0.5,
        contrastive_weight: float = 1.0,
        contrastive_temp: float = 0.07,
        initial_gumbel_temp: float = 1.0,
        temp_annealing_rate: float = 0.999,
        min_gumbel_temp: float = 0.1,
        project_up: Optional[bool] = None,
    ):
        super().__init__()
        self.weight_decay = weight_decay
        self.lm = LanguageModel(base_model_name)
        self.clm_loss = CLMLoss()
        if contrastive_weight > 0:
            self.contrastive_loss = InfoNCELoss(
                do_late_interaction=do_late_interaction,
                do_distance=do_distance,
                exp_decay=exp_decay,
                alpha=alpha,
                temperature=contrastive_temp,
                seq_len=seq_len,
            )
        self.clm_weight = clm_weight
        self.contrastive_weight = contrastive_weight
        self.gumbel_temp = initial_gumbel_temp
        self.temp_annealing_rate = temp_annealing_rate
        self.optim_name = optim_name
        self.batch_size = batch_size
        self.min_gumbel_temp = min_gumbel_temp
        self.dropout = dropout
        self.lr = lr
        if project_up is not None and contrastive_weight > 0:
            self._configure_ffn(project_up)
        elif contrastive_weight > 0:
            self.fc1 = nn.Linear(self.lm.hidden_size, self.lm.hidden_size * 4)
            self.fc2 = nn.Linear(self.lm.hidden_size * 4, self.lm.hidden_size)

    def _calculate_losses(self, batch: Dict[str, torch.Tensor]):
        q_embs, q_logits = self(batch["q_input_ids"], batch["q_attention_mask"])
        k_embs, _ = self(batch["k_input_ids"], batch["k_attention_mask"])

        clm_loss = self.clm_loss(
            q_logits, batch["q_input_ids"], batch["q_attention_mask"]
        )

        if self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss(
                q_embs,
                k_embs,
                batch["author_label"],
                batch["q_attention_mask"],
                batch["k_attention_mask"],
                gumbel_temp=self.gumbel_temp,
            )
        else:
            contrastive_loss = 0.0

        total_loss = (
            self.clm_weight * clm_loss + self.contrastive_weight * contrastive_loss
        )
        return total_loss, clm_loss, contrastive_loss

    def _configure_ffn(self, project_up: bool):
        if project_up:
            self.fc1 = nn.Linear(self.lm.hidden_size, self.lm.hidden_size * 4)
            self.fc2 = nn.Linear(self.lm.hidden_size * 4, self.lm.hidden_size * 4)
        else:
            self.fc1 = nn.Linear(self.lm.hidden_size, self.lm.hidden_size)
            self.fc2 = nn.Linear(self.lm.hidden_size, self.lm.hidden_size)

    def configure_optimizers(self):
        optimizer = self.optim_map[self.optim_name](
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states, logits = self.lm(input_ids, attention_mask)
        if self.contrastive_weight > 0:
            embs = F.dropout(hidden_states, p=self.dropout, training=self.training)
            embs = F.gelu(self.fc1(embs))
            embs = F.dropout(embs, p=self.dropout, training=self.training)
            embs = self.fc2(embs)
        else:
            embs = hidden_states
        return embs, logits

    def training_step(self, batch, batch_idx: int):
        total_loss, clm_loss, contrastive_loss = self._calculate_losses(batch)
        self.log("gumbel_temp", self.gumbel_temp, prog_bar=True)
        self.log_dict(
            {
                "train/total_loss": total_loss,
                "train/clm_loss": clm_loss,
                "train/contrastive_loss": contrastive_loss,
            },
            prog_bar=True,
        )
        # Anneal temperature each step
        if self.gumbel_temp > self.min_gumbel_temp:
            self.gumbel_temp *= self.temp_annealing_rate
        return total_loss

    def validation_step(self, batch, batch_idx: int):
        total_loss, clm_loss, contrastive_loss = self._calculate_losses(batch)
        self.log_dict(
            {
                "val/total_loss": total_loss,
                "val/clm_loss": clm_loss,
                "val/contrastive_loss": contrastive_loss,
            },
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx: int):
        total_loss, clm_loss, contrastive_loss = self._calculate_losses(batch)
        self.log_dict(
            {
                "test/total_loss": total_loss,
                "test/clm_loss": clm_loss,
                "test/contrastive_loss": contrastive_loss,
            }
        )
