# deep_stylometry/modules/modeling_style_embedding.py

import logging
from typing import Dict, Optional

import lightning as L
import torch
import torch.nn.functional as F
from torcheval.metrics import BinaryAUROC, HitRate, ReciprocalRank
from transformers import AutoModel, get_linear_schedule_with_warmup

from deep_stylometry.modules.contrastive_loss import ContrastiveLoss


class StyleEmbedding(L.LightningModule):

    def __init__(
        self,
        base_model_name: str = "AnnaWegmann/Style-Embedding",
        margin: float = 0.5,
        batch_size: int = 8,
        lr: float = 2e-05,
        weight_decay=1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Training
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

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
        self.lm = AutoModel.from_pretrained(base_model_name)
        self.contrastive_loss = ContrastiveLoss(margin=margin)

    def _compute_losses(self, batch: Dict[str, torch.Tensor]):
        q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        k_embs = self(
            input_ids=batch["k_input_ids"],
            attention_mask=batch["k_attention_mask"],
        )

        all_scores, pos_query_scores, pos_query_targets, loss = self.contrastive_loss(
            query_embs=q_embs,
            key_embs=k_embs,
            labels=batch["author_label"],
            q_mask=batch["attention_mask"],
            k_mask=batch["k_attention_mask"],
        )

        metrics = {
            "all_scores": all_scores,
            "pos_query_scores": pos_query_scores,
            "pos_query_targets": pos_query_targets,
            "total_loss": loss,
        }

        return metrics

    def configure_optimizers(self):  # type: ignore[override]
        logging.info(
            f"""Configuring optimizer: {self.optim_name} with lr={self.lr},"""
            """weight_decay={self.weight_decay}"""
        )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-08,
        )

        # Calculate steps dynamically
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.1 * total_steps))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
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
        out = self.lm(input_ids, attention_mask=attention_mask)
        embs = out[0]

        return embs

    def training_step(self, batch, batch_idx: int):
        metrics = self._compute_losses(batch)

        self.log(
            "train_total_loss",
            metrics["total_loss"],
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
            preds = metrics["all_scores"].diag()
            pos_targets = metrics["pos_query_targets"]
            targets = batch["author_label"]

            self.val_auroc.update(preds, targets)
            self.val_hr1.update(pos_preds, pos_targets)
            self.val_hr5.update(pos_preds, pos_targets)
            self.val_hr10.update(pos_preds, pos_targets)
            self.val_rr.update(pos_preds, pos_targets)

        self.log(
            "val_total_loss",
            metrics["total_loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def on_validation_epoch_end(self):
        auroc = self.val_auroc.compute().to(self.device)
        avg_hr1 = self.val_hr1.compute().mean().to(self.device)
        avg_hr5 = self.val_hr5.compute().mean().to(self.device)
        avg_hr10 = self.val_hr10.compute().mean().to(self.device)
        mrr = self.val_rr.compute().mean().to(self.device)
        self.log_dict(
            {
                "val_auroc": auroc.item(),
                "val_hr1": avg_hr1.item(),
                "val_hr5": avg_hr5.item(),
                "val_hr10": avg_hr10.item(),
                "val_mrr": mrr.item(),
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
            preds = metrics["all_scores"].diag()
            pos_targets = metrics["pos_query_targets"]
            targets = batch["author_label"]

            self.test_auroc.update(preds, targets)
            self.test_hr1.update(pos_preds, pos_targets)
            self.test_hr5.update(pos_preds, pos_targets)
            self.test_hr10.update(pos_preds, pos_targets)
            self.test_rr.update(pos_preds, pos_targets)

        self.log(
            "test_total_loss",
            metrics["total_loss"],
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
                "test_auroc": auroc.item(),
                "test_hr1": avg_hr1.item(),
                "test_hr5": avg_hr5.item(),
                "test_hr10": avg_hr10.item(),
                "test_mrr": mrr.item(),
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
