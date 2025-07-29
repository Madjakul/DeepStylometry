# deep_stylometry/modules/modeling_deep_stylometry.py

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torcheval.metrics import BinaryAUROC, HitRate, ReciprocalRank
from transformers import get_cosine_schedule_with_warmup

from deep_stylometry.modules.hard_margin_loss import HardMarginLoss
from deep_stylometry.modules.hybrid_loss import HybridLoss
from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.modules.language_model import LanguageModel
from deep_stylometry.modules.margin_loss import MarginLoss
from deep_stylometry.modules.triplet_loss import TripletLoss

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class DeepStylometry(L.LightningModule):
    """DeepStylometry is a PyTorch Lightning module that implements a deep
    stylometry model for text representation learning. It combines a language
    model with a contrastive loss function to learn embeddings that capture
    stylistic features of text. The model supports various loss functions,
    including InfoNCE, triplet loss, hybrid loss, hard margin loss, and margin
    loss.

    Parameters
    ----------
    cfg : BaseConfig
        Configuration object containing model and execution parameters, including
        the loss function to be used, learning rate, weight decay, and other training
        arguments.

    Attributes
    ----------
    cfg : BaseConfig
        Configuration object with model and execution parameters.
    gumbel_temp : float
        Initial Gumbel temperature for the contrastive loss, used to control the
        sharpness of the softmax distribution.
    contrastive_loss : nn.Module
        The contrastive loss function used for training the model. It can be one of
        InfoNCELoss, TripletLoss, HybridLoss, HardMarginLoss, or MarginLoss,
        depending on the configuration.
    val_auroc : BinaryAUROC
        AUROC metric for validation, used to evaluate the model's performance on
        distinguishing between positive and negative samples.
    val_hr1 : HitRate
        Hit rate at k=1 for validation, measuring the proportion of times the
        correct positive sample is ranked first among the negative samples.
    val_hr5 : HitRate
        Hit rate at k=5 for validation, measuring the proportion of times the
        correct positive sample is ranked within the top 5 among the negative samples.
    val_hr10 : HitRate
        Hit rate at k=10 for validation, measuring the proportion of times the
        correct positive sample is ranked within the top 10 among the negative samples.
    val_rr : ReciprocalRank
        Reciprocal rank for validation, measuring the average rank of the correct
        positive sample across all validation batches.
    test_auroc : BinaryAUROC
        AUROC metric for testing, used to evaluate the model's performance on
        distinguishing between positive and negative samples in the test set.
    test_hr1 : HitRate
        Hit rate at k=1 for testing, measuring the proportion of times the
        correct positive sample is ranked first among the negative samples in the test
        set.
    test_hr5 : HitRate
        Hit rate at k=5 for testing, measuring the proportion of times the
        correct positive sample is ranked within the top 5 among the negative samples
        in the test set.
    test_hr10 : HitRate
        Hit rate at k=10 for testing, measuring the proportion of times the
        correct positive sample is ranked within the top 10 among the negative samples
        in the test set.
    test_rr : ReciprocalRank
        Reciprocal rank for testing, measuring the average rank of the correct
        positive sample across all test batches.
    lm : LanguageModel
        The language model used for generating text embeddings. It can be a pre-trained
        transformer model or a custom model defined in the configuration.
    fc1 : nn.Linear
        Optional linear layer for projecting the embeddings to a different space,
        used if `add_linear_layers` is set to True in the configuration.
    fc2 : nn.Linear
        Optional linear layer for further projecting the embeddings, used if
        `add_linear_layers` is set to True in the configuration.
    loss_map : Dict[str, nn.Module]
        A mapping of loss function names to their corresponding classes. This allows
        for easy selection of the loss function based on the configuration.
    """

    loss_map = {
        "info_nce": InfoNCELoss,
        "triplet": TripletLoss,
        "hybrid": HybridLoss,
        "hard_margin": HardMarginLoss,
        "margin": MarginLoss,
    }
    val_auroc: BinaryAUROC
    val_hr1: HitRate
    val_hr5: HitRate
    val_hr10: HitRate
    val_rr: ReciprocalRank
    test_auroc: BinaryAUROC
    test_hr1: HitRate
    test_hr5: HitRate
    test_hr10: HitRate
    test_rr: ReciprocalRank

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.gumbel_temp = self.cfg.model.initial_gumbel_temp
        self.contrastive_loss = self.loss_map[cfg.execution.loss](cfg)

        # Model
        self.lm = LanguageModel(cfg)
        if self.cfg.model.add_linear_layers:
            hidden_size = self.lm.hidden_size
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)

    def _compute_losses(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses for a batch of data. This method computes the
        language model loss and the contrastive loss using the embeddings
        generated by the language model. It also handles the positive and
        negative samples for contrastive learning.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A dictionary containing the input tensors for the model, including
            `input_ids`, `attention_mask`, `pos_input_ids`, `pos_attention_mask`,
            `neg_input_ids`, and `neg_attention_mask`. The `labels` key is optional
            and only exists if the MLM collator is used.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the computed losses and metrics:
            - `lm_loss`: The language model loss.
            - `contrastive_loss`: The contrastive loss.
            - `total_loss`: The total loss, combining the language model loss and
              the contrastive loss.
            - `all_scores`: The similarity scores for all query-key pairs.
            - `targets`: The target indices for the positive samples.
            - `poss`: The positive scores for the positive samples.
            - `negs`: The negative scores for the negative samples.
        """
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

        k_embs = torch.cat([pos_embs, neg_embs], dim=0)  # (2B, S, H)
        k_mask = torch.cat(
            [batch["pos_attention_mask"], batch["neg_attention_mask"]],
            dim=0,
        )  # (2B, S)

        loss_metrics = self.contrastive_loss(
            query_embs=q_embs,
            key_embs=k_embs,
            q_mask=batch["attention_mask"],
            k_mask=k_mask,
            gumbel_temp=self.gumbel_temp,
        )

        total_loss = (self.cfg.execution.lm_loss_weight * lm_loss) + loss_metrics[
            "loss"
        ]

        metrics = {
            "all_scores": loss_metrics["all_scores"],
            "targets": loss_metrics["targets"],
            "poss": loss_metrics["poss"],
            "negs": loss_metrics["negs"],
            "contrastive_loss": loss_metrics["loss"],
            "lm_loss": lm_loss * self.cfg.execution.lm_loss_weight,
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
            lr=self.cfg.execution.lr,  # type: ignore
            weight_decay=self.cfg.execution.weight_decay,  # type: ignore
            betas=self.cfg.execution.betas,  # type: ignore
            eps=self.cfg.execution.eps,  # type: ignore
        )
        # Calculate steps dynamically
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.1 * total_steps))

        if (
            self.cfg.model.auto_anneal_gumbel
            and self.cfg.model.initial_gumbel_temp is not None
        ):
            total_temp_range = (  # type: ignore
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
        """Override the optimizer_step method to include Gumbel temperature
        annealing.

        This method is called after each optimizer step to update the
        Gumbel temperature if auto-annealing is enabled. The temperature
        is decreased linearly based on the configured delta until it
        reaches the minimum temperature specified in the configuration.
        """
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        # Update Gumbel temperature after each optimizer step
        if (
            self.cfg.model.auto_anneal_gumbel
            and self.cfg.model.initial_gumbel_temp is not None
        ):
            new_temp = self.gumbel_temp - self.gumbel_linear_delta  # type: ignore
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
            projected_embs = self.fc2(embs)
        else:
            projected_embs = last_hidden_states

        return lm_loss, last_hidden_states, projected_embs

    def training_step(self, batch, batch_idx: int) -> Float[torch.Tensor, ""]:
        metrics = self._compute_losses(batch)

        # Log metrics
        if self.cfg.model.initial_gumbel_temp is not None:
            self.log(
                "gumbel_temp",
                self.gumbel_temp,  # type: ignore
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

    def on_validation_start(self):
        """Move validation metrics to correct device before validation."""
        self.val_auroc = BinaryAUROC(device=self.device)
        self.val_hr1 = HitRate(k=1, device=self.device)
        self.val_hr5 = HitRate(k=5, device=self.device)
        self.val_hr10 = HitRate(k=10, device=self.device)
        self.val_rr = ReciprocalRank(device=self.device)

    def validation_step(self, batch, batch_idx: int) -> None:
        metrics = self._compute_losses(batch)
        all_scores = metrics["all_scores"]
        targets = metrics["targets"]
        poss = metrics["poss"]
        negs = metrics["negs"]
        batch_size = targets.size(0)
        binary_scores = torch.cat([poss, negs], dim=0)
        labels = torch.cat(
            [torch.ones(batch_size), torch.zeros(batch_size)], dim=0
        ).long()

        self.val_auroc.update(binary_scores, labels)
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
        auroc = self.val_auroc.compute()
        avg_hr1 = self.val_hr1.compute().mean()
        avg_hr5 = self.val_hr5.compute().mean()
        avg_hr10 = self.val_hr10.compute().mean()
        mrr = self.val_rr.compute().mean()
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

    def on_test_start(self) -> None:
        # this.current_device is now available
        self.test_auroc = BinaryAUROC(device=self.device)
        self.test_hr1 = HitRate(k=1, device=self.device)
        self.test_hr5 = HitRate(k=5, device=self.device)
        self.test_hr10 = HitRate(k=10, device=self.device)
        self.test_rr = ReciprocalRank(device=self.device)

    def test_step(self, batch, batch_idx: int) -> None:
        metrics = self._compute_losses(batch)
        all_scores = metrics["all_scores"]
        targets = metrics["targets"]
        poss = metrics["poss"]
        negs = metrics["negs"]
        batch_size = targets.size(0)
        binary_scores = torch.cat([poss, negs], dim=0)
        labels = torch.cat(
            [torch.ones(batch_size), torch.zeros(batch_size)], dim=0
        ).long()

        self.test_auroc.update(binary_scores, labels)
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
        auroc = self.test_auroc.compute()
        avg_hr1 = self.test_hr1.compute().mean()
        avg_hr5 = self.test_hr5.compute().mean()
        avg_hr10 = self.test_hr10.compute().mean()
        mrr = self.test_rr.compute().mean()
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
