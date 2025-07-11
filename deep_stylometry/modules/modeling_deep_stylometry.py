# deep_stylometry/modules/modeling_deep_stylometry.py

import logging
from typing import TYPE_CHECKING, Dict, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import HitRate, MulticlassAUROC, ReciprocalRank
from transformers import get_cosine_schedule_with_warmup

from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.modules.language_model import LanguageModel
from deep_stylometry.modules.triplet_loss import TripletLoss

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class DeepStylometry(L.LightningModule):

    loss_map = {"info_nce": InfoNCELoss, "triplet": TripletLoss}

    def __init__(self, cfg: "BaseConfig"):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.gumbel_temp = self.cfg.model.initial_gumbel_temp
        self.contrastive_loss = self.loss_map[cfg.train.loss](cfg)

        # Validation metrics
        self.val_auroc = MulticlassAUROC(num_classes=3 * self.cfg.data.batch_size).to(
            self.device
        )
        self.val_hr1 = HitRate(k=1).to(self.device)
        self.val_hr5 = HitRate(k=5).to(self.device)
        self.val_hr10 = HitRate(k=10).to(self.device)
        self.val_rr = ReciprocalRank().to(self.device)

        # Test metrics
        self.test_auroc = MulticlassAUROC(num_classes=3 * self.cfg.data.batch_size).to(
            self.device
        )
        self.test_hr1 = HitRate(k=1).to(self.device)
        self.test_hr5 = HitRate(k=5).to(self.device)
        self.test_hr10 = HitRate(k=10).to(self.device)
        self.test_rr = ReciprocalRank().to(self.device)

        # Model
        self.lm = LanguageModel(cfg)
        if self.add_linear_layers:
            hidden_size = self.lm.hidden_size
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)

    def _compute_losses(self, batch: Dict[str, torch.Tensor]):
        lm_loss, _, q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels", None),  # only exists if MLM collator is used
        )
        _, _, pos_embs = self(
            input_ids=batch["pos_input_ids"],
            attention_mask=batch["pos_attention_mask"],
        )
        _, _, neg_embs = self(
            input_ids=batch["neg_input_ids"],
            attention_mask=batch["neg_attention_mask"],
        )

        k_embs = torch.cat([q_embs, pos_embs, neg_embs], dim=0)  # (3B, S, H)
        k_mask = torch.cat(
            [
                batch["attention_mask"],
                batch["pos_attention_mask"],
                batch["neg_attention_mask"],
            ],
            dim=0,
        )  # (3B, S)

        all_scores, targets, contrastive_loss = self.contrastive_loss(
            query_embs=q_embs,
            key_embs=k_embs,
            q_mask=batch["attention_mask"],
            k_mask=k_mask,
            gumbel_temp=self.gumbel_temp,
        )

        total_loss = (self.lm_weight * lm_loss) + contrastive_loss

        metrics = {
            "all_scores": all_scores,
            "targets": targets,
            "lm_loss": lm_loss * self.lm_weight,
            "contrastive_loss": contrastive_loss,
            "total_loss": total_loss,
        }

        return metrics

    def configure_optimizers(self):  # type: ignore[override]
        logging.info(
            f"""Configuring optimizer: AdamW with lr={self.lr},"""
            f"""" weight_decay={self.weight_decay}, betas={self.betas},"""
            f""" eps={self.eps}."""
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            betas=self.cfg.train.betas,
            eps=self.cfg.train.eps,
        )
        # Calculate steps dynamically
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.1 * total_steps))

        if (
            self.cfg.model.auto_anneal_gumbel
            and self.cfg.model.initial_gumbel_temp is not None
        ):
            total_temp_range = (
                self.cfg.model.initial_gumbel_temp - self.cfg.model.min_gumbel_temp
            )
            self.gumbel_linear_delta = total_temp_range / total_steps

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=self.cfg.train.num_cycles,
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
        if (
            self.cfg.model.auto_anneal_gumbel
            and self.cfg.model.initial_gumbel_temp is not None
        ):
            new_temp = self.gumbel_temp - self.gumbel_linear_delta
            self.gumbel_temp = max(new_temp, self.cfg.model.min_gumbel_temp)
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

        if self.linear_layers:
            embs = F.dropout(last_hidden_states, p=self.cfg.model.dropout)
            embs = F.gelu(self.fc1(embs))
            embs = F.dropout(embs, p=self.cfg.model.dropout)
            projected_embs = self.fc2(embs) + last_hidden_states  # residual
        else:
            projected_embs = last_hidden_states

        return lm_loss, last_hidden_states, projected_embs

    def training_step(self, batch, batch_idx: int):
        metrics = self._compute_losses(batch)

        # Log metrics
        if self.initial_gumbel_temp is not None:
            self.log(
                "gumbel_temp",
                self.gumbel_temp,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=self.cfg.data.batch_size,
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
            batch_size=self.cfg.data.batch_size,
        )
        return metrics["total_loss"]

    def validation_step(self, batch, batch_idx: int):
        metrics = self._compute_losses(batch)

        all_scores = metrics["all_scores"]
        targets = metrics["targets"]

        self.val_auroc.update(all_scores, targets)
        self.val_hr1.update(all_scores, targets)
        self.val_hr5.update(all_scores, targets)
        self.val_hr10.update(all_scores, targets)
        self.val_rr.update(all_scores, targets)

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
            batch_size=self.cfg.data.batch_size,
        )

    def on_validation_epoch_end(self):
        self.log("completed_epoch", self.current_epoch, prog_bar=False)
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

        all_scores = metrics["all_scores"]
        targets = metrics["targets"]

        self.test_auroc.update(all_scores, targets)
        self.test_hr1.update(all_scores, targets)
        self.test_hr5.update(all_scores, targets)
        self.test_hr10.update(all_scores, targets)
        self.test_rr.update(all_scores, targets)

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
            batch_size=self.cfg.data.batch_size,
        )

    def on_test_epoch_end(self):
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
