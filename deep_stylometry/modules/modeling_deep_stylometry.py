# deep_stylometry/modules/modeling_deep_stylometry.py

import logging
from typing import TYPE_CHECKING, Any, Dict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from deep_stylometry.modules.alignment_uniformity_loss import \
    AlignmentUniformityLoss
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
        self.contrastive_loss = self.loss_map[cfg.train.loss](cfg)

        assert cfg.model.expansion_ratio > 0, "expansion_ratio must be > 0"

        # Model
        self.lm = LanguageModel(cfg)
        hidden_size = self.lm.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(cfg.model.dropout),
            nn.Linear(hidden_size, hidden_size * cfg.model.expansion_ratio),
            nn.ReLU(),
            nn.Linear(hidden_size * cfg.model.expansion_ratio, hidden_size),
        )

        # Other
        self.alignment_uniformity_loss = AlignmentUniformityLoss()

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
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
        embs = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(embs)

    def training_step(self, batch, batch_idx):
        # Local Forward Pass
        q_embs = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"], attention_mask=batch["pos_attention_mask"]
        )
        neg_embs = self(
            input_ids=batch["neg_input_ids"], attention_mask=batch["neg_attention_mask"]
        )

        q_mask = batch["attention_mask"]

        max_seq = max(pos_embs.size(1), neg_embs.size(1))

        # Pad embeddings
        pos_embs = F.pad(pos_embs, (0, 0, 0, max_seq - pos_embs.size(1)))
        neg_embs = F.pad(neg_embs, (0, 0, 0, max_seq - neg_embs.size(1)))

        # Pad masks
        pos_mask = F.pad(
            batch["pos_attention_mask"],
            (0, max_seq - batch["pos_attention_mask"].size(1)),
        )
        neg_mask = F.pad(
            batch["neg_attention_mask"],
            (0, max_seq - batch["neg_attention_mask"].size(1)),
        )

        if self.trainer.world_size > 1 and self.cfg.train.gather:

            # Use pos_embs and pos_mask (NOT batch["pos_attention_mask"])
            global_pos_embs = self.gather_with_padding(pos_embs, pad_value=0.0)
            global_pos_mask = self.gather_with_padding(pos_mask, pad_value=0)

            # Use neg_embs and neg_mask (NOT batch["neg_attention_mask"])
            global_neg_embs = self.gather_with_padding(neg_embs, pad_value=0.0)
            global_neg_mask = self.gather_with_padding(neg_mask, pad_value=0)

            # Construct Keys
            k_embs = torch.cat([global_pos_embs, global_neg_embs], dim=0)
            k_mask = torch.cat([global_pos_mask, global_neg_mask], dim=0)

            # Targets Offset Calculation
            local_bs = q_embs.size(0)
            global_offset = self.trainer.global_rank * local_bs
            targets = torch.arange(local_bs, device=self.device) + global_offset

        else:
            # Single GPU Logic (No extra padding needed here anymore)
            k_embs = torch.cat([pos_embs, neg_embs], dim=0)
            k_mask = torch.cat([pos_mask, neg_mask], dim=0)
            targets = torch.arange(q_embs.size(0), device=self.device)

        loss_metrics = self.contrastive_loss(
            query_embs=q_embs,
            key_embs=k_embs,
            q_mask=q_mask,
            k_mask=k_mask,
            targets=targets,
            q_input_ids=batch["input_ids"],
        )

        self.log("train/loss", loss_metrics["loss"], prog_bar=True)
        return loss_metrics["loss"]

    def validation_step(self, batch, batch_idx):
        # Local Forward Pass
        q_embs = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"], attention_mask=batch["pos_attention_mask"]
        )
        neg_embs = self(
            input_ids=batch["neg_input_ids"], attention_mask=batch["neg_attention_mask"]
        )

        q_mask = batch["attention_mask"]

        # Single GPU Logic
        max_seq = max(pos_embs.size(1), neg_embs.size(1))
        pos_embs = F.pad(pos_embs, (0, 0, 0, max_seq - pos_embs.size(1)))
        neg_embs = F.pad(neg_embs, (0, 0, 0, max_seq - neg_embs.size(1)))
        pos_mask = F.pad(
            batch["pos_attention_mask"],
            (0, max_seq - batch["pos_attention_mask"].size(1)),
        )
        neg_mask = F.pad(
            batch["neg_attention_mask"],
            (0, max_seq - batch["neg_attention_mask"].size(1)),
        )
        k_embs = torch.cat([pos_embs, neg_embs], dim=0)
        k_mask = torch.cat([pos_mask, neg_mask], dim=0)
        targets = torch.arange(q_embs.size(0), device=self.device)

        alignment_uniformity_metrics = self.alignment_uniformity_loss(
            query_embs=q_embs,
            key_embs=k_embs,
            q_mask=q_mask,
            k_mask=k_mask,
            targets=targets,
        )
        self.log_dict(
            {
                "val/alignment_loss": alignment_uniformity_metrics["alignment_loss"],
                "val/uniformity_loss": alignment_uniformity_metrics["uniformity_loss"],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg.data.batch_size,
        )

    def gather_with_padding(
        self, local_tensor: torch.Tensor, pad_value: float = 0
    ) -> torch.Tensor:
        """
        Robust Gather:
        1. Identifies the global maximum sequence length across all GPUs.
        2. Pads the local tensor to that global max.
        3. Gathers and concatenates.
        """
        if self.trainer.world_size == 1:
            return local_tensor

        # 1. Get local max sequence length (dim 1 is seq_len)
        local_max_len = torch.tensor(local_tensor.shape[1], device=self.device)

        # 2. Sync to find global max length
        global_max_len = local_max_len.clone()
        self.trainer.strategy.reduce(global_max_len, reduce_op="max")

        # 3. Pad locally if necessary
        diff = global_max_len.item() - local_max_len.item()
        if diff > 0:
            # F.pad logic: (pad_last_dim_left, pad_last_dim_right, pad_2nd_last_left, ...)
            if local_tensor.dim() == 3:  # Embeddings (Batch, Seq, Hidden)
                pad_config = (0, 0, 0, diff)
            elif local_tensor.dim() == 2:  # Masks or IDs (Batch, Seq)
                pad_config = (0, diff)
            else:
                raise ValueError(f"Unexpected tensor shape: {local_tensor.shape}")

            # Use the specific pad_value provided (0 for masks/embs, pad_token_id for input_ids)
            local_tensor = F.pad(local_tensor, pad_config, value=pad_value)

        # 4. Standard all_gather
        gathered = self.all_gather(local_tensor, sync_grads=True)

        # 5. Flatten [World, Batch, ...] -> [GlobalBatch, ...]
        return gathered.flatten(0, 1)
