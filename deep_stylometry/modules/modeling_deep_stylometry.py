# deep_stylometry/modules/modeling_deep_stylometry.py

import logging
from typing import Dict, Optional

import bitsandbytes as bnb
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from transformers.models.blip.modeling_blip import contrastive_loss

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
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        seq_len: int,
        is_decoder_model: bool,
        lr: float = 2e-5,
        dropout: float = 0.1,
        weight_decay: float = 1e-2,
        lm_weight: float = 1.0,
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
        self.save_hyperparameters()  # Save hyperparameters
        self.weight_decay = weight_decay
        self.lm = LanguageModel(base_model_name, is_decoder_model)
        self.tokenizer = tokenizer
        self.is_decoder_model = is_decoder_model
        self.lm_weight = lm_weight
        self.contrastive_weight = contrastive_weight
        self.gumbel_temp = initial_gumbel_temp
        self.temp_annealing_rate = temp_annealing_rate
        self.optim_name = optim_name
        self.batch_size = batch_size
        self.min_gumbel_temp = min_gumbel_temp
        self.dropout = dropout
        self.lr = lr

        # Projection layers setup
        if contrastive_weight > 0:
            self.contrastive_loss = InfoNCELoss(
                do_late_interaction=do_late_interaction,
                do_distance=do_distance,
                exp_decay=exp_decay,
                alpha=alpha,
                temperature=contrastive_temp,
                seq_len=seq_len,
            )
            hidden_size = self.lm.hidden_size
            if project_up is True:
                self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
                self.fc2 = nn.Linear(hidden_size * 4, hidden_size * 4)
            elif project_up is False:
                self.fc1 = nn.Linear(hidden_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
            else:  # Default projection if project_up is None but contrastive_weight > 0
                self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
                self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

    def _compute_losses(self, batch: Dict[str, torch.Tensor]):
        lm_loss, _, q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels", None),
        )
        _, _, k_embs = self(
            input_ids=batch["k_input_ids"],
            attention_mask=batch["k_attention_mask"],
        )

        contrastive_loss = 0.0
        if self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss(
                q_embs,
                k_embs,
                batch["author_label"],
                batch["attention_mask"],  # Use original attention masks
                batch["k_attention_mask"],
                gumbel_temp=self.gumbel_temp,
            )

        total_loss = (self.lm_weight * lm_loss) + (
            self.contrastive_weight * contrastive_loss
        )

        return (
            lm_loss * self.lm_weight,
            contrastive_loss * self.contrastive_weight,
            total_loss,
        )

    def configure_optimizers(self):
        logging.info(
            f"Configuring optimizer: {self.optim_name} with lr={self.lr}, weight_decay={self.weight_decay}"
        )

        optimizer = self.optim_map[self.optim_name](
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Add scheduler here if needed
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [scheduler]
        return optimizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        lm_loss, last_hidden_states = self.lm(
            input_ids, attention_mask=attention_mask, labels=labels
        )

        if self.contrastive_weight > 0:
            embs = F.dropout(last_hidden_states, p=self.dropout, training=self.training)
            embs = F.gelu(self.fc1(embs))
            embs = F.dropout(embs, p=self.dropout, training=self.training)
            projected_embs = self.fc2(embs)
        else:
            projected_embs = last_hidden_states

        return lm_loss, last_hidden_states, projected_embs

    def training_step(self, batch, batch_idx: int):
        lm_loss, contrastive_loss, total_loss = self._compute_losses(batch)

        # Log metrics
        self.log(
            "gumbel_temp",
            self.gumbel_temp,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        self.log(
            "train/total_loss",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train/lm_loss",
            lm_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        if self.contrastive_weight > 0:
            self.log(
                "train/contrastive_loss",
                contrastive_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )

        # Anneal Gumbel temperature (if used by contrastive loss)
        # Check if self.contrastive_loss exists and has gumbel temp logic before annealing
        if hasattr(self, "contrastive_loss") and hasattr(
            self.contrastive_loss, "temperature"
        ):  # Simple check
            if self.gumbel_temp > self.min_gumbel_temp:
                self.gumbel_temp *= self.temp_annealing_rate

        return total_loss

    # Add validation_step and test_step if needed, mirroring _calculate_losses logic
    # Ensure to use appropriate logging keys (e.g., "val/total_loss")

    # def validation_step(self, batch, batch_idx: int):
    #     total_loss, lm_loss, contrastive_loss = self._calculate_losses(batch)
    #
    #     self.log(
    #         "val/total_loss",
    #         total_loss,
    #         prog_bar=True,
    #         on_step=False,
    #         on_epoch=True,
    #         batch_size=self.batch_size,
    #     )
    #     self.log(
    #         "val/lm_loss",
    #         lm_loss,
    #         prog_bar=False,
    #         on_step=False,
    #         on_epoch=True,
    #         batch_size=self.batch_size,
    #     )
    #     if self.contrastive_weight > 0:
    #         self.log(
    #             "val/contrastive_loss",
    #             contrastive_loss,
    #             prog_bar=False,
    #             on_step=False,
    #             on_epoch=True,
    #             batch_size=self.batch_size,
    #         )
    #     return total_loss

    def test_step(self, batch, batch_idx: int):
        total_loss, lm_loss, contrastive_loss = self._calculate_losses(batch)

        self.log(
            "test/total_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test/lm_loss",
            lm_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        if self.contrastive_weight > 0:
            self.log(
                "test/contrastive_loss",
                contrastive_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        return total_loss
