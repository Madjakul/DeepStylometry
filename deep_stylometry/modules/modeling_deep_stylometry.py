# deep_stylometry/modules/modeling_deep_stylometry.py

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torcheval.metrics import HitRate, MulticlassAUROC, ReciprocalRank
from transformers import get_cosine_schedule_with_warmup

from deep_stylometry.modules.hybrid_loss import HybridLoss
from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.modules.language_model import LanguageModel
from deep_stylometry.modules.triplet_loss import TripletLoss

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class DeepStylometry(L.LightningModule):

    loss_map = {"info_nce": InfoNCELoss, "triplet": TripletLoss, "hybrid": HybridLoss}

    def __init__(self, cfg: "BaseConfig"):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.gumbel_temp = self.cfg.model.initial_gumbel_temp
        self.contrastive_loss = self.loss_map[cfg.execution.loss](cfg)

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
        if self.cfg.model.add_linear_layers:
            hidden_size = self.lm.hidden_size
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)

    def _compute_losses(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
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

        total_loss = (self.cfg.execution.lm_loss_weight * lm_loss) + contrastive_loss

        metrics = {
            "all_scores": all_scores,
            "targets": targets,
            "lm_loss": lm_loss * self.cfg.execution.lm_loss_weight,
            "contrastive_loss": contrastive_loss,
            "total_loss": total_loss,
        }

        return metrics

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore[override]
        logging.info(
            f"""Configuring optimizer: AdamW with lr={self.cfg.execution.lr},
             weight_decay={self.cfg.execution.weight_decay},
             betas={self.cfg.execution.betas}, eps={self.cfg.execution.eps}."""
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.execution.lr,
            weight_decay=self.cfg.execution.weight_decay,
            betas=self.cfg.execution.betas,
            eps=self.cfg.execution.eps,
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
            num_cycles=self.cfg.execution.num_cycles,
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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure) -> None:  # type: ignore[override]
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

        if self.cfg.model.add_linear_layers:
            embs = F.layer_norm(
                last_hidden_states,
                normalized_shape=(self.lm.hidden_size,),
                weight=None,
                bias=None,
                eps=1e-5,
            )
            embs = F.dropout(embs, p=self.cfg.model.dropout, training=self.training)
            embs = F.relu(self.fc1(embs))
            embs = F.dropout(embs, p=self.cfg.model.dropout, training=self.training)
            projected_embs = self.fc2(embs)  # NO residual
        else:
            projected_embs = last_hidden_states

        return lm_loss, last_hidden_states, projected_embs

    def training_step(self, batch, batch_idx: int) -> Float[torch.Tensor, ""]:
        metrics = self._compute_losses(batch)

        # Log metrics
        if self.cfg.model.initial_gumbel_temp is not None:
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

    def validation_step(self, batch, batch_idx: int) -> None:
        metrics = self._compute_losses(batch)

        all_scores = metrics["all_scores"]
        targets = metrics["targets"]

        # --- START: PADDING LOGIC ---

        # Get the expected number of classes from the metric itself
        num_classes = self.val_auroc.num_classes  # This will be 32

        # Get the shape of the current predictions
        current_batch_size, current_num_classes = all_scores.shape

        # If the current number of classes doesn't match the expected, pad it.
        # This will only happen on the last, smaller batch.
        if current_num_classes != num_classes:
            # Create a new tensor with the correct shape [16, 32] and fill with a large negative value
            padded_preds = torch.full(
                (current_batch_size, num_classes),
                -torch.inf,  # Use -inf to ensure these are ranked last
                device=all_scores.device,
                dtype=all_scores.dtype,
            )

            # Copy the actual predictions into the new padded tensor
            padded_preds[:, :current_num_classes] = all_scores

            # Use the padded tensor for the update
            all_scores = padded_preds

        # --- END: PADDING LOGIC ---

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

    def on_validation_epoch_end(self) -> None:
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

    def test_step(self, batch, batch_idx: int) -> None:
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

    def on_test_epoch_end(self) -> None:
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
