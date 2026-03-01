# deep_stylometry/callbacks/retrieval_eval_callback.py

import logging
from typing import List

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
)

K = 100
Q_CHUNK = 256
K_CHUNK = 256


class RetrievalEvalCallback(L.Callback):
    """Evaluates dense retrieval during validation.

    Always uses mean pooling to isolate training objective effect.
    """

    def __init__(self) -> None:
        super().__init__()

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
        li = getattr(pl_module.contrastive_loss, "pool", None)
        if not isinstance(li, LateInteraction):
            logging.warning("LateInteraction not found; using mean pooling for eval.")
            pool = MeanInteraction()
        else:
            logging.info("LateInteraction found for eval.")
            pool = li

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

        top_scores = torch.full((n_queries, K), float("-inf"))
        top_indices = torch.zeros((n_queries, K), dtype=torch.long)

        for q_start in tqdm(
            range(0, n_queries, Q_CHUNK), desc="Evaluating Retrieval", unit="query"
        ):
            q_end = min(q_start + Q_CHUNK, n_queries)

            # 1. Move Query chunks to GPU
            q_chunk = q_embs[q_start:q_end].to(pl_module.device)
            q_mask_chunk = q_masks[q_start:q_end].to(pl_module.device)

            for k_start in range(0, n_corpus, K_CHUNK):
                k_end = min(k_start + K_CHUNK, n_corpus)

                # 2. Move Key chunks to GPU
                k_chunk = k_embs[k_start:k_end].to(pl_module.device)
                k_mask_chunk = k_masks[k_start:k_end].to(pl_module.device)

                # Compute interaction (GPU math) and immediately move back to CPU
                chunk_scores = pool(
                    query_embs=q_chunk,
                    key_embs=k_chunk,
                    q_mask=q_mask_chunk,
                    k_mask=k_mask_chunk,
                ).cpu()  # (q_chunk, k_chunk)

                combined_scores = torch.cat(
                    [top_scores[q_start:q_end], chunk_scores], dim=1
                )
                combined_indices = torch.cat(
                    [
                        top_indices[q_start:q_end],
                        torch.arange(k_start, k_end)
                        .unsqueeze(0)
                        .expand(q_end - q_start, -1),
                    ],
                    dim=1,
                )

                # Maintain top K safely
                best_scores, best_local = combined_scores.topk(K, dim=1)
                top_scores[q_start:q_end] = best_scores
                top_indices[q_start:q_end] = combined_indices.gather(1, best_local)

        # Build Ranx Run correctly filtering out the -inf padded values
        run = Run(
            {
                f"q{i}": {
                    f"d{int(top_indices[i, j])}": float(top_scores[i, j])
                    for j in range(K)
                    if top_scores[i, j] != float("-inf")
                }
                for i in range(n_queries)
            }
        )

        for k in [5, 10, 20, 100]:
            metrics = evaluate_run(hard_qrels, soft_qrels, run, k)

            # Log your metrics
            pl_module.log_dict(
                {f"val/{metric_name}": v for metric_name, v in metrics.items()},
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
