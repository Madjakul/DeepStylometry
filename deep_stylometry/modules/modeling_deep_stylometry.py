# deep_stylometry/modules/modeling_deep_stylometry.py

import logging
from typing import Dict, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import HitRate, MulticlassAUROC, ReciprocalRank
from transformers import get_cosine_schedule_with_warmup

from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.modules.language_model import LanguageModel


class DeepStylometry(L.LightningModule):

    def __init__(
        self,
        base_model_name: str,
        batch_size: int,
        seq_len: int,
        is_decoder_model: bool,
        lr: float = 2e-5,
        dropout: float = 0.1,
        weight_decay: float = 1e-2,
        num_cycles: float = 0.5,
        lm_weight: float = 1.0,
        add_linear_layers: bool = True,
        contrastive_temp: float = 7e-2,
        use_softmax: bool = True,
        pooling_method: str = "mean",
        distance_weightning: str = "none",
        initial_gumbel_temp: float = 1.0,
        auto_anneal_gumbel: bool = True,
        min_gumbel_temp: float = 1e-6,
        alpha: float = 1.0,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Misc
        self.is_decoder_model = is_decoder_model
        self.lm_weight = lm_weight
        self.add_linear_layers = add_linear_layers
        self.initial_gumbel_temp = initial_gumbel_temp
        self.gumbel_temp = initial_gumbel_temp
        self.min_gumbel_temp = min_gumbel_temp
        self.auto_anneal_gumbel = auto_anneal_gumbel

        # Training
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_cycles = num_cycles
        self.dropout = dropout
        self.lr = lr
        self.betas = betas
        self.eps = eps

        # Validation metrics
        self.val_auroc = MulticlassAUROC(num_classes=self.batch_size).to(self.device)
        self.val_hr1 = HitRate(k=1).to(self.device)
        self.val_hr5 = HitRate(k=5).to(self.device)
        self.val_hr10 = HitRate(k=10).to(self.device)
        self.val_rr = ReciprocalRank().to(self.device)

        # Test metrics
        self.test_auroc = MulticlassAUROC(num_classes=self.batch_size).to(self.device)
        self.test_hr1 = HitRate(k=1).to(self.device)
        self.test_hr5 = HitRate(k=5).to(self.device)
        self.test_hr10 = HitRate(k=10).to(self.device)
        self.test_rr = ReciprocalRank().to(self.device)

        # Model
        self.lm = LanguageModel(base_model_name, is_decoder_model)
        self.contrastive_loss = InfoNCELoss(
            alpha=alpha,
            temperature=contrastive_temp,
            seq_len=seq_len,
            use_softmax=use_softmax,
            pooling_method=pooling_method,
            distance_weightning=distance_weightning,
        )
        hidden_size = self.lm.hidden_size
        if self.add_linear_layers:
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)

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

        total_loss = (self.lm_weight * lm_loss) + contrastive_loss

        metrics = {
            "all_scores": all_scores,
            "pos_query_scores": pos_query_scores,
            "pos_query_targets": pos_query_targets,
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
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
        )
        # Calculate steps dynamically
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.1 * total_steps))

        if self.auto_anneal_gumbel and self.initial_gumbel_temp is not None:
            total_temp_range = self.initial_gumbel_temp - self.min_gumbel_temp
            self.gumbel_linear_delta = total_temp_range / total_steps

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=self.num_cycles,
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
        if self.auto_anneal_gumbel and self.initial_gumbel_temp is not None:
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

        if self.linear_layers:
            embs = F.dropout(last_hidden_states, p=self.dropout, training=self.training)
            embs = F.gelu(self.fc1(embs))
            embs = F.dropout(embs, p=self.dropout, training=self.training)
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

        if metrics["pos_query_scores"] is not None:
            pos_preds = metrics["pos_query_scores"]
            pos_targets = metrics["pos_query_targets"]

            # --- START: PADDING LOGIC ---

            # Get the expected number of classes from the metric itself
            num_classes = self.val_auroc.num_classes  # This will be 32

            # Get the shape of the current predictions
            current_batch_size, current_num_classes = pos_preds.shape

            # If the current number of classes doesn't match the expected, pad it.
            # This will only happen on the last, smaller batch.
            if current_num_classes != num_classes:
                # Create a new tensor with the correct shape [16, 32] and fill with a large negative value
                padded_preds = torch.full(
                    (current_batch_size, num_classes),
                    -torch.inf,  # Use -inf to ensure these are ranked last
                    device=pos_preds.device,
                    dtype=pos_preds.dtype,
                )

                # Copy the actual predictions into the new padded tensor
                padded_preds[:, :current_num_classes] = pos_preds

                # Use the padded tensor for the update
                pos_preds = padded_preds

            # --- END: PADDING LOGIC ---

            self.val_auroc.update(pos_preds, pos_targets)
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
        if metrics["pos_query_scores"] is not None:
            pos_preds = metrics["pos_query_scores"]
            pos_targets = metrics["pos_query_targets"]

            # --- START: PADDING LOGIC ---

            # Get the expected number of classes from the metric itself
            num_classes = self.val_auroc.num_classes  # This will be 32

            # Get the shape of the current predictions
            current_batch_size, current_num_classes = pos_preds.shape

            # If the current number of classes doesn't match the expected, pad it.
            # This will only happen on the last, smaller batch.
            if current_num_classes != num_classes:
                # Create a new tensor with the correct shape [16, 32] and fill with a large negative value
                padded_preds = torch.full(
                    (current_batch_size, num_classes),
                    -torch.inf,  # Use -inf to ensure these are ranked last
                    device=pos_preds.device,
                    dtype=pos_preds.dtype,
                )

                # Copy the actual predictions into the new padded tensor
                padded_preds[:, :current_num_classes] = pos_preds

                # Use the padded tensor for the update
                pos_preds = padded_preds

            # --- END: PADDING LOGIC ---

            self.test_auroc.update(pos_preds, pos_targets)
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
