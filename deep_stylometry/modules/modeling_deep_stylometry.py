# deep_stylometry/modules/modeling_deep_stylometry.py

import logging
from typing import Dict, Optional

import bitsandbytes as bnb
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (BinaryAUROC, BinaryF1Score,
                                         BinaryPrecision, BinaryRecall)
from transformers import get_cosine_schedule_with_warmup

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
        is_decoder_model: bool,
        lr: float = 2e-5,
        dropout: float = 0.1,
        weight_decay: float = 1e-2,
        lm_weight: float = 1.0,
        do_late_interaction: bool = False,
        use_max: bool = True,
        do_distance: bool = False,
        exp_decay: bool = False,
        alpha: float = 0.5,
        contrastive_weight: float = 1.0,
        contrastive_temp: float = 0.07,
        initial_gumbel_temp: float = 1.0,
        temp_annealing_rate: Optional[float] = 1e-3,
        min_gumbel_temp: float = 1e-5,
        project_up: Optional[bool] = None,
        auto_anneal_gumbel: Optional[bool] = None,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters
        self.weight_decay = weight_decay
        self.lm = LanguageModel(base_model_name, is_decoder_model)
        self.is_decoder_model = is_decoder_model
        self.lm_weight = lm_weight
        self.contrastive_weight = contrastive_weight
        self.initial_gumbel_temp = initial_gumbel_temp
        self.gumbel_temp = initial_gumbel_temp
        self.temp_annealing_rate = temp_annealing_rate
        self.optim_name = optim_name
        self.batch_size = batch_size
        self.min_gumbel_temp = min_gumbel_temp
        self.auto_anneal_gumbel = auto_anneal_gumbel
        self.dropout = dropout
        self.lr = lr
        # Validation metrics
        self.val_auroc = BinaryAUROC(thresholds=None)
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        # Test metrics
        self.test_auroc = BinaryAUROC(thresholds=None)
        self.test_f1 = BinaryF1Score()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        if contrastive_weight > 0:
            self.contrastive_loss = InfoNCELoss(
                do_late_interaction=do_late_interaction,
                do_distance=do_distance,
                exp_decay=exp_decay,
                alpha=alpha,
                temperature=contrastive_temp,
                seq_len=seq_len,
                use_max=use_max,
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
        pos_query_scores = None
        pos_query_targets = None
        if self.contrastive_weight > 0:
            pos_query_scores, pos_query_targets, contrastive_loss = (
                self.contrastive_loss(
                    q_embs,
                    k_embs,
                    batch["author_label"],
                    batch["attention_mask"],  # Use original attention masks
                    batch["k_attention_mask"],
                    gumbel_temp=self.gumbel_temp,
                )
            )

        total_loss = (self.lm_weight * lm_loss) + (
            self.contrastive_weight * contrastive_loss
        )

        metrics = {
            "pos_query_scores": pos_query_scores,
            "pos_query_targets": pos_query_targets,
            "lm_loss": lm_loss * self.lm_weight,
            "contrastive_loss": contrastive_loss * self.contrastive_weight,
            "total_loss": total_loss,
        }

        return metrics

    def configure_optimizers(self):  # type: ignore[override]
        logging.info(
            f"Configuring optimizer: {self.optim_name} with lr={self.lr}, weight_decay={self.weight_decay}"
        )

        optimizer = self.optim_map[self.optim_name](
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Calculate steps dynamically
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.05 * total_steps))

        if self.auto_anneal_gumbel:
            # Automatically anneal Gumbel temperature based on training steps
            self.temp_annealing_rate = (
                self.min_gumbel_temp / self.initial_gumbel_temp
            ) ** (1 / total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,  # 0.5 cosine cycle → single smooth decay
            last_epoch=-1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

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
        metrics = self._compute_losses(batch)

        # Log metrics
        self.log(
            "gumbel_temp",
            self.gumbel_temp,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        self.log_dict(
            {
                "train_total_loss": metrics["total_loss"],
                "train_lm_loss": metrics["lm_loss"],
                "train_contrastive_loss": metrics["contrastive_loss"],
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        # Anneal Gumbel temperature (if used by contrastive loss)
        if hasattr(self, "contrastive_loss") and hasattr(
            self.contrastive_loss, "do_late_interaction"
        ):
            # Anneal Gumbel temperature
            self.gumbel_temp = max(
                self.gumbel_temp * self.temp_annealing_rate, self.min_gumbel_temp
            )

        return metrics["total_loss"]

    def validation_step(self, batch, batch_idx: int):
        metrics = self._compute_losses(batch)

        if self.contrastive_weight > 0 and metrics["pos_query_scores"] is not None:
            pos_query_scores = metrics["pos_query_scores"]
            pos_query_targets = metrics["pos_query_targets"]

            # Generate binary labels (1 for correct key, 0 otherwise)
            binary_labels = torch.zeros_like(pos_query_scores, dtype=torch.long)
            rows = torch.arange(
                pos_query_scores.size(0), device=pos_query_scores.device
            )
            binary_labels[rows, pos_query_targets] = 1

            # Flatten scores and labels
            flat_scores = pos_query_scores.flatten()
            flat_labels = binary_labels.flatten()

            # Update metrics
            self.val_auroc(flat_scores, flat_labels)
            self.val_f1(flat_scores, flat_labels)
            self.val_precision(flat_scores, flat_labels)
            self.val_recall(flat_scores, flat_labels)

        self.log_dict(
            {
                "val_total_loss": metrics["total_loss"],
                "val_lm_loss": metrics["lm_loss"],
                "val_contrastive_loss": metrics["contrastive_loss"],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "val_auroc": self.val_auroc.compute(),
                "val_f1": self.val_f1.compute(),
                "val_precision": self.val_precision.compute(),
                "val_recall": self.val_recall.compute(),
            },
            prog_bar=True,
        )
        self.val_auroc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def test_step(self, batch, batch_idx: int):
        metrics = self._compute_losses(batch)
        if self.contrastive_weight > 0 and metrics["pos_query_scores"] is not None:
            pos_query_scores = metrics["pos_query_scores"]
            pos_query_targets = metrics["pos_query_targets"]

            # Generate binary labels (1 for correct key, 0 otherwise)
            binary_labels = torch.zeros_like(pos_query_scores, dtype=torch.long)
            rows = torch.arange(
                pos_query_scores.size(0), device=pos_query_scores.device
            )
            binary_labels[rows, pos_query_targets] = 1

            # Flatten scores and labels
            flat_scores = pos_query_scores.flatten()
            flat_labels = binary_labels.flatten()

            # Update metrics
            self.test_auroc(flat_scores, flat_labels)
            self.test_f1(flat_scores, flat_labels)
            self.test_precision(flat_scores, flat_labels)
            self.test_recall(flat_scores, flat_labels)

        self.log_dict(
            {
                "test_total_loss": metrics["total_loss"],
                "test_lm_loss": metrics["lm_loss"],
                "test_contrastive_loss": metrics["contrastive_loss"],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def on_test_epoch_end(self):
        self.log_dict(
            {
                "test_auroc": self.test_auroc.compute(),
                "test_f1": self.test_f1.compute(),
                "test_precision": self.test_precision.compute(),
                "test_recall": self.test_recall.compute(),
            },
            prog_bar=True,
        )
        self.test_auroc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
