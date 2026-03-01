# deep_stylometry/callbacks/retrieval_eval_callback.py

import logging
from typing import TYPE_CHECKING, Dict, List

import lightning as L
import torch
from ranx import Run
from tqdm import tqdm

from deep_stylometry.modules.late_interaction import LateInteraction
from deep_stylometry.modules.mean_interaction import MeanInteraction
from deep_stylometry.utils.eval_utils import (
    build_corpus,
    build_qrels,
    evaluate_run,
    gather_targets,
    pad_and_cat_1d,
)

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class RetrievalEvalCallback(L.Callback):
    """Evaluates dense retrieval during validation.

    Always uses mean pooling to isolate training objective effect.
    """

    def __init__(
        self,
        cfg: "BaseConfig",
        k: int = 100,
        shortlist_k: int = 500,
        chunk_size: int = 8192,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.k = k
        self.shortlist_k = shortlist_k
        self.chunk_size = chunk_size
        self.mean_pool = MeanInteraction()
        self.li = LateInteraction(self.cfg)
        self._reset()

    def _reset(self) -> None:
        self.q_embs: List[torch.Tensor] = []
        self.q_masks: List[torch.Tensor] = []
        self.q_ids: List[torch.Tensor] = []
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
        self.q_ids.append(batch["input_ids"].cpu())
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

        # q_ids needs to match q_embs seq dim
        q_ids = pad_and_cat_1d(self.q_ids, target_seq=q_embs.size(1))

        n_queries = q_embs.size(0)
        n_corpus = k_embs.size(0)
        targets = gather_targets(self.target_indices)
        hard_qrels, soft_qrels = build_qrels(n_queries, n_corpus, targets)

        logging.info("Starting Stage 1: Dense Retrieval (Mean Pooling)...")
        self.mean_pool = self.mean_pool.to(pl_module.device)

        dense_top_scores = torch.full((n_queries, self.shortlist_k), float("-inf"))
        dense_top_indices = torch.zeros((n_queries, self.shortlist_k), dtype=torch.long)

        for q_start in tqdm(
            range(0, n_queries, self.chunk_size), desc="Dense Eval", unit="chunk"
        ):
            q_end = min(q_start + self.chunk_size, n_queries)
            q_chunk = q_embs[q_start:q_end].to(pl_module.device)
            q_mask_chunk = q_masks[q_start:q_end].to(pl_module.device)

            for k_start in range(0, n_corpus, self.chunk_size):
                k_end = min(k_start + self.chunk_size, n_corpus)
                k_chunk = k_embs[k_start:k_end].to(pl_module.device)
                k_mask_chunk = k_masks[k_start:k_end].to(pl_module.device)

                chunk_scores = self.mean_pool(
                    query_embs=q_chunk,
                    key_embs=k_chunk,
                    q_mask=q_mask_chunk,
                    k_mask=k_mask_chunk,
                ).cpu()

                combined_scores = torch.cat(
                    [dense_top_scores[q_start:q_end], chunk_scores], dim=1
                )
                combined_indices = torch.cat(
                    [
                        dense_top_indices[q_start:q_end],
                        torch.arange(k_start, k_end)
                        .unsqueeze(0)
                        .expand(q_end - q_start, -1),
                    ],
                    dim=1,
                )

                best_scores, best_local = combined_scores.topk(self.shortlist_k, dim=1)
                dense_top_scores[q_start:q_end] = best_scores
                dense_top_indices[q_start:q_end] = combined_indices.gather(
                    1, best_local
                )

        # Build and Evaluate Dense Run (Extracting up to self.k for metrics)
        eval_k = min(self.k, self.shortlist_k)
        dense_run = Run(
            {
                f"q{i}": {
                    f"d{int(dense_top_indices[i, j])}": float(dense_top_scores[i, j])
                    for j in range(eval_k)
                    if dense_top_scores[i, j] != float("-inf")
                }
                for i in range(n_queries)
            }
        )

        for k_val in [5, 10, 20, 100]:
            if k_val <= eval_k:
                metrics = evaluate_run(hard_qrels, soft_qrels, dense_run, k_val)
                pl_module.log_dict(
                    {f"val/single_{m}": v for m, v in metrics.items()},
                    on_epoch=True,
                )
                if k_val == eval_k:
                    logging.info(f"Dense @{k_val}: {metrics}")

        logging.info("Starting Stage 2: Late Interaction Reranking...")
        ts_run_dict: Dict[str, Dict[str, float]] = {}

        # Loop query by query, fetching ONLY its shortlist candidates
        for i in tqdm(range(n_queries), desc="LI Reranking", unit="query"):
            cand_ids = dense_top_indices[i]  # Shape: (SHORTLIST_K,)

            # Prepare Query (Shape: 1, S_q, H)
            q_emb_i = q_embs[i].unsqueeze(0).to(pl_module.device)
            q_mask_i = q_masks[i].unsqueeze(0).to(pl_module.device)
            q_ids_i = q_ids[i].unsqueeze(0).to(pl_module.device)

            all_scores = []

            # Chunk the LI computation across the shortlist candidates
            for start in range(0, self.shortlist_k, self.chunk_size):
                end = min(start + self.chunk_size, self.shortlist_k)
                chunk_cand_ids = cand_ids[start:end]

                cand_embs_chunk = k_embs[chunk_cand_ids].to(pl_module.device)
                cand_masks_chunk = k_masks[chunk_cand_ids].to(pl_module.device)

                chunk_scores = self.li(
                    query_embs=q_emb_i,
                    key_embs=cand_embs_chunk,
                    q_mask=q_mask_i,
                    k_mask=cand_masks_chunk,
                    q_input_ids=q_ids_i,
                ).cpu()  # Output shape: (1, chunk_size)

                all_scores.append(chunk_scores.squeeze(0))

            all_scores = torch.cat(all_scores, dim=0)
            topk_scores, topk_local = all_scores.topk(eval_k)

            ts_run_dict[f"q{i}"] = {
                f"d{int(cand_ids[topk_local[j]])}": float(topk_scores[j])
                for j in range(topk_local.size(0))
            }

        ts_run = Run(ts_run_dict)

        for k_val in [5, 10, 20, 100]:
            if k_val <= eval_k:
                metrics = evaluate_run(hard_qrels, soft_qrels, ts_run, k_val)
                pl_module.log_dict(
                    {f"val/multi_{m}": v for m, v in metrics.items()},
                    on_epoch=True,
                )
                if k_val == eval_k:
                    logging.info(f"Two-stage @{k_val}: {metrics}")
