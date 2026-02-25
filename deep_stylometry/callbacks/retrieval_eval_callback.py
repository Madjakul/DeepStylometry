# deep_stylometry/callbacks/retrieval_eval_callback.py

import logging
from typing import List

import lightning as L
import torch

from deep_stylometry.modules.mean_interaction import MeanInteraction
from deep_stylometry.utils.eval_utils import (
    build_corpus,
    build_qrels,
    evaluate_run,
    gather_targets,
    scores_to_run,
)


class RetrievalEvalCallback(L.Callback):
    """Evaluates dense retrieval during validation.

    Always uses mean pooling to isolate training objective effect.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean_pool = MeanInteraction()
        self._reset()

    def _reset(self) -> None:
        self.q_embs: List[torch.Tensor] = []
        self.q_masks: List[torch.Tensor] = []
        self.pos_embs: List[torch.Tensor] = []
        self.pos_masks: List[torch.Tensor] = []
        self.neg_embs: List[torch.Tensor] = []
        self.neg_masks: List[torch.Tensor] = []
        self.target_indices: List[torch.Tensor] = []

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._reset()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        q = pl_module(batch["input_ids"], batch["attention_mask"]).cpu()
        p = pl_module(batch["pos_input_ids"], batch["pos_attention_mask"]).cpu()
        n = pl_module(batch["neg_input_ids"], batch["neg_attention_mask"]).cpu()

        self.q_embs.append(q)
        self.q_masks.append(batch["attention_mask"].cpu())
        self.pos_embs.append(p)
        self.pos_masks.append(batch["pos_attention_mask"].cpu())
        self.neg_embs.append(n)
        self.neg_masks.append(batch["neg_attention_mask"].cpu())

        if "target_indices" in batch:
            self.target_indices.append(batch["target_indices"].cpu())

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.q_embs:
            return

        q_embs, q_masks, k_embs, k_masks = build_corpus(
            self.q_embs,
            self.q_masks,
            self.pos_embs,
            self.pos_masks,
            self.neg_embs,
            self.neg_masks,
        )

        n_queries = q_embs.size(0)
        n_corpus = k_embs.size(0)
        targets = gather_targets(self.target_indices)
        hard_qrels, soft_qrels = build_qrels(n_queries, n_corpus, targets)

        scores = self.mean_pool(
            query_embs=q_embs, key_embs=k_embs, q_mask=q_masks, k_mask=k_masks
        )

        run = scores_to_run(scores, 100)
        metrics = evaluate_run(hard_qrels, soft_qrels, run, 100)
        pl_module.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=pl_module.cfg.data.batch_size,
        )
        logging.info(f"Retrieval eval: {metrics}")

        run = scores_to_run(scores, 20)
        metrics = evaluate_run(hard_qrels, soft_qrels, run, 20)
        pl_module.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=pl_module.cfg.data.batch_size,
        )
        logging.info(f"Retrieval eval: {metrics}")

        run = scores_to_run(scores, 10)
        metrics = evaluate_run(hard_qrels, soft_qrels, run, 10)
        pl_module.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=pl_module.cfg.data.batch_size,
        )
        logging.info(f"Retrieval eval: {metrics}")

        run = scores_to_run(scores, 5)
        metrics = evaluate_run(hard_qrels, soft_qrels, run, 5)
        pl_module.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=pl_module.cfg.data.batch_size,
        )
        logging.info(f"Retrieval eval: {metrics}")
