# deep_stylometry/modules/modeling_deep_stylometry.py

import logging
from typing import Dict, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import BinaryAUROC, HitRate, ReciprocalRank
from transformers import get_cosine_schedule_with_warmup

from deep_stylometry.modules.contrastive_loss import ContrastiveLoss
from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.modules.language_model import LanguageModel
from deep_stylometry.optimizers import SOAP, SophiaG


class DeepStylometry(L.LightningModule):

    optim_map = {
        "adamw": torch.optim.AdamW,
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
        contrastive_weight: float = 1.0,
        contrastive_temp: float = 7e-2,
        do_late_interaction: bool = True,
        use_max: bool = False,
        initial_gumbel_temp: float = 1.0,
        auto_anneal_gumbel: bool = True,
        gumbel_linear_delta: float = 1e-3,
        min_gumbel_temp: float = 1e-6,
        do_distance: bool = True,
        exp_decay: bool = True,
        alpha: float = 1.0,
        project_up: Optional[bool] = None,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters

        # Misc
        self.is_decoder_model = is_decoder_model
        self.lm_weight = lm_weight
        self.contrastive_weight = contrastive_weight
        self.initial_gumbel_temp = initial_gumbel_temp
        self.gumbel_temp = initial_gumbel_temp
        self.gumbel_linear_delta = gumbel_linear_delta
        self.min_gumbel_temp = min_gumbel_temp
        self.auto_anneal_gumbel = auto_anneal_gumbel

        # Training
        self.weight_decay = weight_decay
        self.optim_name = optim_name.lower()
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr

        # Validation metrics
        self.val_auroc = BinaryAUROC().to(self.device)
        self.val_hr1 = HitRate(k=1).to(self.device)
        self.val_hr5 = HitRate(k=5).to(self.device)
        self.val_hr10 = HitRate(k=10).to(self.device)
        self.val_rr = ReciprocalRank().to(self.device)

        # Test metrics
        self.test_auroc = BinaryAUROC().to(self.device)
        self.test_hr1 = HitRate(k=1).to(self.device)
        self.test_hr5 = HitRate(k=5).to(self.device)
        self.test_hr10 = HitRate(k=10).to(self.device)
        self.test_rr = ReciprocalRank().to(self.device)

        # Model
        self.lm = LanguageModel(base_model_name, is_decoder_model)
        if contrastive_weight > 0:
            self.contrastive_loss = InfoNCELoss(  # ContrastiveLoss(
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
            # Default projection if project_up is None but contrastive_weight > 0
            else:
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
        all_scores = None
        pos_query_scores = None
        pos_query_targets = None
        if self.contrastive_weight > 0:
            all_scores, pos_query_scores, pos_query_targets, contrastive_loss = (
                self.contrastive_loss(
                    q_embs,
                    k_embs,
                    batch["author_label"],
                    batch["attention_mask"],
                    batch["k_attention_mask"],
                    gumbel_temp=self.gumbel_temp,
                )
            )

        total_loss = (self.lm_weight * lm_loss) + (
            self.contrastive_weight * contrastive_loss
        )

        metrics = {
            "all_scores": all_scores,
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
        warmup_steps = max(1, int(0.1 * total_steps))

        if self.auto_anneal_gumbel:
            total_temp_range = self.initial_gumbel_temp - self.min_gumbel_temp
            self.gumbel_linear_delta = total_temp_range / total_steps

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,  # 0.5 cosine cycle -> single smooth decay
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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):  # type: ignore[override]
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        # Update Gumbel temperature after each optimizer step
        if self.auto_anneal_gumbel:
            new_temp = self.gumbel_temp - self.gumbel_linear_delta
            self.gumbel_temp = max(new_temp, self.min_gumbel_temp)
            self.log("gumbel_temp", self.gumbel_temp, prog_bar=True)

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
            logger=True,
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
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return metrics["total_loss"]

    def validation_step(self, batch, batch_idx: int):
        metrics = self._compute_losses(batch)

        if self.contrastive_weight > 0 and metrics["pos_query_scores"] is not None:
            pos_preds = F.softmax(metrics["pos_query_scores"], dim=-1)
            preds = F.softmax(metrics["all_scores"], dim=-1).diag()
            pos_targets = metrics["pos_query_targets"]
            targets = batch["author_label"]

            self.val_auroc.update(preds, targets)
            self.val_hr1.update(pos_preds, pos_targets)
            self.val_hr5.update(pos_preds, pos_targets)
            self.val_hr10.update(pos_preds, pos_targets)
            self.val_rr.update(pos_preds, pos_targets)

        self.log_dict(
            {
                "val_total_loss": metrics["total_loss"],
                "val_lm_loss": metrics["lm_loss"],
                "val_contrastive_loss": metrics["contrastive_loss"],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def on_validation_epoch_end(self):
        if self.contrastive_weight > 0:
            auroc = self.val_auroc.compute().to(self.device)
            avg_hr1 = self.val_hr1.compute().mean().to(self.device)
            avg_hr5 = self.val_hr5.compute().mean().to(self.device)
            avg_hr10 = self.val_hr10.compute().mean().to(self.device)
            mrr = self.val_rr.compute().mean().to(self.device)
            self.log_dict(
                {
                    "val_auroc": auroc,
                    "val_hr1": avg_hr1,
                    "val_hr5": avg_hr5,
                    "val_hr10": avg_hr10,
                    "val_mrr": mrr,
                },
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.val_auroc.reset()
            self.val_hr1.reset()
            self.val_hr5.reset()
            self.val_hr10.reset()
            self.val_rr.reset()

    def test_step(self, batch, batch_idx: int):
        metrics = self._compute_losses(batch)
        if self.contrastive_weight > 0 and metrics["pos_query_scores"] is not None:
            pos_preds = F.softmax(metrics["pos_query_scores"], dim=-1)
            preds = F.softmax(metrics["all_scores"], dim=-1).diag()
            pos_targets = metrics["pos_query_targets"]
            targets = batch["author_label"]

            self.test_auroc.update(preds, targets)
            self.test_hr1.update(pos_preds, pos_targets)
            self.test_hr5.update(pos_preds, pos_targets)
            self.test_hr10.update(pos_preds, pos_targets)
            self.test_rr.update(pos_preds, pos_targets)

        self.log_dict(
            {
                "test_total_loss": metrics["total_loss"],
                "test_lm_loss": metrics["lm_loss"],
                "test_contrastive_loss": metrics["contrastive_loss"],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def on_test_epoch_end(self):
        if self.contrastive_weight > 0:
            auroc = self.test_auroc.compute().to(self.device)
            avg_hr1 = self.test_hr1.compute().mean().to(self.device)
            avg_hr5 = self.test_hr5.compute().mean().to(self.device)
            avg_hr10 = self.test_hr10.compute().mean().to(self.device)
            mrr = self.test_rr.compute().mean().to(self.device)
            self.log_dict(
                {
                    "test_auroc": auroc,
                    "test_hr1": avg_hr1,
                    "test_hr5": avg_hr5,
                    "test_hr10": avg_hr10,
                    "test_mrr": mrr,
                },
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
