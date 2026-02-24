# deep_stylometry/callbacks/test_eval_callback.py

import logging
from typing import Dict, List

import lightning as L
import torch
from ranx import Run

from deep_stylometry.modules.mean_interaction import MeanInteraction
from deep_stylometry.utils.eval_utils import (
    build_corpus,
    pad_and_cat_1d,
    build_qrels,
    scores_to_run,
    gather_targets,
    evaluate_run,
)


class TestEvalCallback(L.Callback):
    """
    Test callback: Dense retrieval + optional two-stage (dense shortlist
    + LateInteraction reranking). BM25 is in bm25_evaluator.py.
    """

    def __init__(
        self,
        k: int = 100,
        shortlist_k: int = 500,
        run_two_stage: bool = True,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.k = k
        self.shortlist_k = shortlist_k
        self.run_two_stage = run_two_stage
        self.chunk_size = chunk_size
        self.mean_pool = MeanInteraction()
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

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._reset()

    def on_test_batch_end(
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

    def on_test_epoch_end(self, trainer, pl_module) -> None:
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

        # 1. Dense retrieval
        dense_scores = self.mean_pool(
            query_embs=q_embs, key_embs=k_embs, q_mask=q_masks, k_mask=k_masks
        )
        dense_run = scores_to_run(dense_scores, self.k)
        dense_metrics = evaluate_run(hard_qrels, soft_qrels, dense_run, self.k)
        pl_module.log_dict(
            {f"test/dense/{k}": v for k, v in dense_metrics.items()}, on_epoch=True
        )
        logging.info(f"Dense: {dense_metrics}")

        # 2. Two-stage
        if not self.run_two_stage:
            return

        from deep_stylometry.modules.late_interaction import LateInteraction

        li = getattr(pl_module.contrastive_loss, "pool", None)
        if not isinstance(li, LateInteraction):
            logging.warning("LateInteraction not found; skipping two-stage eval.")
            return

        # Move to CPU for reranking
        original_device = next(
            (p.device for p in li.parameters()),
            next((b.device for _, b in li.named_buffers()), None),
        )
        li_cpu = li.cpu()
        li_cpu.eval()

        shortlist_k = min(self.shortlist_k, n_corpus)
        _, shortlist_indices = dense_scores.topk(shortlist_k, dim=1)

        run_dict: Dict[str, Dict[str, float]] = {}
        for i in range(n_queries):
            cand_ids = shortlist_indices[i]
            cand_embs = k_embs[cand_ids]  # (K, S_k, H)
            cand_masks = k_masks[cand_ids]  # (K, S_k)

            q_emb_i = q_embs[i].unsqueeze(0)  # (1, S_q, H)
            q_mask_i = q_masks[i].unsqueeze(0)  # (1, S_q)
            q_ids_i = q_ids[i].unsqueeze(0)  # (1, S_q)

            all_scores = []
            for start in range(0, shortlist_k, self.chunk_size):
                end = min(start + self.chunk_size, shortlist_k)
                chunk_scores = li_cpu(
                    query_embs=q_emb_i,
                    key_embs=cand_embs[start:end],
                    q_mask=q_mask_i,
                    k_mask=cand_masks[start:end],
                    q_input_ids=q_ids_i,
                )  # (1, chunk)
                all_scores.append(chunk_scores.squeeze(0))

            all_scores = torch.cat(all_scores, dim=0)
            topk_scores, topk_local = all_scores.topk(min(self.k, shortlist_k))
            run_dict[f"q{i}"] = {
                f"d{int(cand_ids[topk_local[j]])}": float(topk_scores[j])
                for j in range(topk_local.size(0))
            }

        ts_run = Run(run_dict)
        ts_metrics = evaluate_run(hard_qrels, soft_qrels, ts_run, self.k)
        pl_module.log_dict(
            {f"test/two_stage/{k}": v for k, v in ts_metrics.items()}, on_epoch=True
        )
        logging.info(f"Two-stage: {ts_metrics}")

        if original_device is not None:
            li.to(original_device)
