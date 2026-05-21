# deep_stylometry/modules/modeling_deep_stylometry.py

from typing import TYPE_CHECKING, Any, Dict, Optional

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
    """Contrastive authorship attribution model built on a pre-trained encoder.

    Wraps a HuggingFace language model with a two-layer MLP projection head
    and a choice of interaction function (mean pooling, full late interaction,
    or patch-level late interaction). Supports optional layer-wise attention
    and mean centering. Trained end-to-end with InfoNCE or triplet loss.

    Parameters
    ----------
    cfg : BaseConfig
        Global configuration object (model, data, and training sub-configs).
    """

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

        # LayerwiseAttention (when pooling_method == "layerwise")
        self.layer_attention: Optional[nn.Module] = None
        if cfg.model.pooling_method == "layerwise":
            from deep_stylometry.modules.layerwise_attention import \
                LayerwiseAttention
            n_layers = self.lm.model.config.num_hidden_layers + 1  # +1 for embedding layer
            self.layer_attention = LayerwiseAttention(n_layers)

        # MeanCenterer (optional, applied during training only)
        self.mean_centerer: Optional[nn.Module] = None
        if getattr(cfg.model, "mean_center", False):
            from deep_stylometry.modules.mean_centerer import MeanCenterer
            self.mean_centerer = MeanCenterer(hidden_size)

        # Other
        self.alignment_uniformity_loss = AlignmentUniformityLoss()

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore[override]
        """Set up AdamW optimizer with cosine learning-rate warmup.

        Warmup covers 10% of the estimated total steps. After warmup the LR
        follows a half-cosine decay to zero.

        Returns
        -------
        dict
            Lightning optimizer/scheduler configuration dict.
        """
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a batch of texts into per-token embeddings.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs, shape ``(batch, seq)``.
        attention_mask : torch.Tensor
            Binary attention mask, shape ``(batch, seq)``.

        Returns
        -------
        torch.Tensor
            Per-token projected embeddings, shape ``(batch, seq, hidden)``.
        """
        if self.cfg.model.pooling_method == "layerwise":
            outputs = self.lm.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # outputs.hidden_states: tuple of (num_hidden_layers + 1) tensors
            combined = self.layer_attention(list(outputs.hidden_states), attention_mask)
            embs = self.head(combined)
        else:
            embs = self.lm(input_ids=input_ids, attention_mask=attention_mask)
            embs = self.head(embs)
        return embs  # (batch, seq, hidden)

    def _pool_and_center(
        self,
        embs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool per-token embeddings then apply mean centering + L2 norm.

        Args:
            embs: Per-token embeddings, shape ``(batch, seq, hidden)``.
            mask: Attention mask, shape ``(batch, seq)``.

        Returns:
            Pooled, centred, L2-normalised embeddings ``(batch, hidden)``.
        """
        mask_f = mask.unsqueeze(-1).float()
        pooled = (embs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
        if self.training:
            self.mean_centerer.update(pooled)
        return self.mean_centerer.apply(pooled)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Compute contrastive loss for one training batch.

        Encodes query, positive, and negative texts; pads to a common sequence
        length; optionally gathers across GPUs for in-batch negatives; and
        computes the configured contrastive loss. When learned PLI is active an
        additional patch-regularisation term is added.

        Returns
        -------
        torch.Tensor
            Scalar loss used for backpropagation.
        """
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

        # Pad input_ids (for patch-level interaction key patching)
        pos_ids = F.pad(
            batch["pos_input_ids"],
            (0, max_seq - batch["pos_input_ids"].size(1)),
        )
        neg_ids = F.pad(
            batch["neg_input_ids"],
            (0, max_seq - batch["neg_input_ids"].size(1)),
        )

        if self.trainer.world_size > 1 and self.cfg.train.gather:

            # Use pos_embs and pos_mask (NOT batch["pos_attention_mask"])
            global_pos_embs = self.gather_with_padding(pos_embs, pad_value=0.0)
            global_pos_mask = self.gather_with_padding(pos_mask, pad_value=0)

            # Use neg_embs and neg_mask (NOT batch["neg_attention_mask"])
            global_neg_embs = self.gather_with_padding(neg_embs, pad_value=0.0)
            global_neg_mask = self.gather_with_padding(neg_mask, pad_value=0)

            # Gather input_ids for patch interaction
            global_pos_ids = self.gather_with_padding(pos_ids, pad_value=0)
            global_neg_ids = self.gather_with_padding(neg_ids, pad_value=0)

            # Construct Keys
            k_embs = torch.cat([global_pos_embs, global_neg_embs], dim=0)
            k_mask = torch.cat([global_pos_mask, global_neg_mask], dim=0)
            k_ids = torch.cat([global_pos_ids, global_neg_ids], dim=0)

            # Targets Offset Calculation
            local_bs = q_embs.size(0)
            global_offset = self.trainer.global_rank * local_bs
            targets = torch.arange(local_bs, device=self.device) + global_offset

        else:
            # Single GPU Logic (No extra padding needed here anymore)
            k_embs = torch.cat([pos_embs, neg_embs], dim=0)
            k_mask = torch.cat([pos_mask, neg_mask], dim=0)
            k_ids = torch.cat([pos_ids, neg_ids], dim=0)
            targets = torch.arange(q_embs.size(0), device=self.device)

        # Apply mean centering if configured (training only)
        if self.mean_centerer is not None:
            q_embs_l = self._pool_and_center(q_embs, q_mask).unsqueeze(1)
            k_embs_l = self._pool_and_center(k_embs, k_mask).unsqueeze(1)
            q_mask_l = torch.ones(q_embs_l.size(0), 1, dtype=torch.long, device=self.device)
            k_mask_l = torch.ones(k_embs_l.size(0), 1, dtype=torch.long, device=self.device)
        else:
            q_embs_l, k_embs_l, q_mask_l, k_mask_l = q_embs, k_embs, q_mask, k_mask

        loss_metrics = self.contrastive_loss(
            query_embs=q_embs_l,
            key_embs=k_embs_l,
            q_mask=q_mask_l,
            k_mask=k_mask_l,
            targets=targets,
            q_input_ids=batch["input_ids"],
            k_input_ids=k_ids,
            step=self.global_step,
        )

        loss = loss_metrics["loss"]

        # Add patch regularisation loss when using learned PLI
        if "patch_reg_loss" in loss_metrics:
            patch_reg = loss_metrics["patch_reg_loss"]
            loss = loss + self.cfg.model.patch_lambda * patch_reg
            self.log("train/patch_reg_loss", patch_reg, prog_bar=False)

        self.log("train/loss", loss_metrics["loss"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """Run one validation batch and log alignment/uniformity and accuracy.

        Does not perform gather across GPUs. Logs ``val/alignment_loss``,
        ``val/uniformity_loss``, ``val/accuracy``, and ``val/loss``.
        """
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
        pos_ids = F.pad(
            batch["pos_input_ids"],
            (0, max_seq - batch["pos_input_ids"].size(1)),
        )
        neg_ids = F.pad(
            batch["neg_input_ids"],
            (0, max_seq - batch["neg_input_ids"].size(1)),
        )
        k_embs = torch.cat([pos_embs, neg_embs], dim=0)
        k_mask = torch.cat([pos_mask, neg_mask], dim=0)
        k_ids = torch.cat([pos_ids, neg_ids], dim=0)
        targets = torch.arange(q_embs.size(0), device=self.device)

        alignment_uniformity_metrics = self.alignment_uniformity_loss(  # type: ignore[call-arg]
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

        # Apply mean centering for val loss (no running-mean update at eval time)
        if self.mean_centerer is not None:
            q_embs_l = self._pool_and_center(q_embs, q_mask).unsqueeze(1)
            k_embs_l = self._pool_and_center(k_embs, k_mask).unsqueeze(1)
            q_mask_l = torch.ones(q_embs_l.size(0), 1, dtype=torch.long, device=self.device)
            k_mask_l = torch.ones(k_embs_l.size(0), 1, dtype=torch.long, device=self.device)
        else:
            q_embs_l, k_embs_l, q_mask_l, k_mask_l = q_embs, k_embs, q_mask, k_mask

        loss_metrics = self.contrastive_loss(
            query_embs=q_embs_l,
            key_embs=k_embs_l,
            q_mask=q_mask_l,
            k_mask=k_mask_l,
            targets=targets,
            q_input_ids=batch["input_ids"],
            k_input_ids=k_ids,
        )

        accuracy = (loss_metrics["poss"] > loss_metrics["negs"]).float().mean()
        self.log(
            "val/accuracy",
            accuracy,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.cfg.data.batch_size,
        )
        self.log(
            "val/loss",
            loss_metrics["loss"],
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.cfg.data.batch_size,
        )

    def test_step(self, batch, batch_idx) -> dict:
        """Encode one test batch and return raw embeddings for offline scoring.

        Returns
        -------
        dict
            Keys: ``q_embs``, ``q_mask``, ``q_input_ids``, ``pos_embs``,
            ``pos_mask``, ``pos_input_ids``, ``neg_embs``, ``neg_mask``,
            ``neg_input_ids``, ``target_indices``. The
            :class:`~deep_stylometry.callbacks.TestEvalCallback` consumes these.
        """
        q_embs = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"], attention_mask=batch["pos_attention_mask"]
        )
        neg_embs = self(
            input_ids=batch["neg_input_ids"], attention_mask=batch["neg_attention_mask"]
        )
        return {
            "q_embs": q_embs,
            "q_mask": batch["attention_mask"],
            "q_input_ids": batch["input_ids"],
            "pos_embs": pos_embs,
            "pos_mask": batch["pos_attention_mask"],
            "pos_input_ids": batch["pos_input_ids"],
            "neg_embs": neg_embs,
            "neg_mask": batch["neg_attention_mask"],
            "neg_input_ids": batch["neg_input_ids"],
            "target_indices": batch.get("target_indices", None),
        }

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
            if local_tensor.dim() == 3:  # Embeddings (Batch, Seq, Hidden)
                pad_config = (0, 0, 0, diff)
            elif local_tensor.dim() == 2:  # Masks or IDs (Batch, Seq)
                pad_config = (0, diff)
            else:
                raise ValueError(f"Unexpected tensor shape: {local_tensor.shape}")

            local_tensor = F.pad(local_tensor, pad_config, value=pad_value)

        # 4. Standard all_gather
        gathered = self.all_gather(local_tensor, sync_grads=True)

        # 5. Flatten [World, Batch, ...] -> [GlobalBatch, ...]
        return gathered.flatten(0, 1)
