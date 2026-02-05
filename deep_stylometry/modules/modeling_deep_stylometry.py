# deep_stylometry/modules/modeling_deep_stylometry.py

import logging
from typing import TYPE_CHECKING, Any, Dict

import lightning as L
import torch
import torch.nn as nn
from jaxtyping import Float
from transformers import get_cosine_schedule_with_warmup

from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.modules.language_model import LanguageModel
from deep_stylometry.modules.triplet_loss import TripletLoss

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class DeepStylometry(L.LightningModule):

    loss_map = {
        "info_nce": InfoNCELoss,
        "triplet": TripletLoss,
    }

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.contrastive_loss = self.loss_map[cfg.execution.loss](cfg)

        assert cfg.model.expansion_ratio > 0, "expansion_ratio must be > 0"

        # Model
        self.lm = LanguageModel(cfg)
        hidden_size = self.lm.hidden_size
        self.model = nn.Sequential(
            self.lm,
            nn.Dropout(cfg.model.dropout),
            nn.Linear(hidden_size, hidden_size * cfg.model.expansion_ratio),
            nn.ReLU(),
            nn.Linear(hidden_size * cfg.model.expansion_ratio, hidden_size),
        )

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore[override]
        logging.info(
            f"""Configuring optimizer: AdamW with lr={self.cfg.execution.lr},
             weight_decay={self.cfg.execution.weight_decay}"""
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.execution.lr,  # type: ignore
            weight_decay=self.cfg.execution.weight_decay,  # type: ignore
        )
        # Calculate steps dynamically
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.1 * total_steps))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx: int) -> Float[torch.Tensor, ""]:
        q_mask = batch["attention_mask"]
        batch_size = q_mask.size(0)

        q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"],
            attention_mask=batch["pos_attention_mask"],
        )
        neg_embs = self(
            input_ids=batch["neg_input_ids"],
            attention_mask=batch["neg_attention_mask"],
        )

        if self.trainer.world_size > 1 and self.cfg.train.gather:
            # all_gather adds a dimension at the start, so we flatten it with the batch dim
            # Shape changes from [num_gpus, batch_size, seq, hidden] -> [global_batch_size, seq, hidden]
            targets = (
                torch.arange(batch_size, device=q_embs.device)
                + batch_size * self.trainer.global_rank
            )
            all_pos_embs = self.trainer.strategy.all_gather(
                pos_embs, sync_grads=True
            ).flatten(0, 1)
            all_pos_mask = self.trainer.strategy.all_gather(
                batch["pos_attention_mask"]
            ).flatten(0, 1)
            all_neg_embs = self.trainer.strategy.all_gather(
                neg_embs, sync_grads=True
            ).flatten(0, 1)
            all_neg_mask = self.trainer.strategy.all_gather(
                batch["neg_attention_mask"]
            ).flatten(0, 1)

            k_embs = torch.cat([all_pos_embs, all_neg_embs], dim=0)
            k_mask = torch.cat([all_pos_mask, all_neg_mask], dim=0)
            loss_metrics = self.contrastive_loss(
                query_embs=q_embs,
                key_embs=k_embs,
                q_mask=q_mask,
                k_mask=k_mask,
                targets=targets,
                q_input_ids=batch["input_ids"],
            )
        else:
            targets = torch.arange(batch_size, device=q_embs.device)
            k_embs = torch.cat([pos_embs, neg_embs], dim=0)
            k_mask = torch.cat(
                [batch["pos_attention_mask"], batch["neg_attention_mask"]], dim=0
            )
            loss_metrics = self.contrastive_loss(
                query_embs=q_embs,
                key_embs=k_embs,
                q_mask=q_mask,
                k_mask=k_mask,
                targets=targets,
                q_input_ids=batch["input_ids"],
            )

        self.log_dict(
            {
                "train/loss": loss_metrics["loss"],
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg.data.batch_size,
        )
        return loss_metrics["loss"]
