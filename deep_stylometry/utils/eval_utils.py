# deep_stylometry/utils/eval_utils.py


from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from ranx import Qrels, Run, evaluate


def build_corpus(
    q_embs_list: List[torch.Tensor],
    q_masks_list: List[torch.Tensor],
    pos_embs_list: List[torch.Tensor],
    pos_masks_list: List[torch.Tensor],
    neg_embs_list: List[torch.Tensor],
    neg_masks_list: List[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad and concatenate per-batch embedding lists into a full corpus.

    Pads all embeddings to the global maximum sequence length across all
    batches and splits, then concatenates positives and negatives into a
    single key tensor.

    Parameters
    ----------
    q_embs_list : list of Tensor
        Per-batch query embeddings ``(B, S_i, H)``.
    q_masks_list : list of Tensor
        Per-batch query attention masks ``(B, S_i)``.
    pos_embs_list : list of Tensor
        Per-batch positive document embeddings.
    pos_masks_list : list of Tensor
        Per-batch positive document masks.
    neg_embs_list : list of Tensor
        Per-batch negative document embeddings.
    neg_masks_list : list of Tensor
        Per-batch negative document masks.

    Returns
    -------
    tuple of Tensor
        ``(q_embs, q_masks, k_embs, k_masks)`` where ``k_embs`` concatenates
        positives followed by negatives along the batch dimension.
    """
    all_embs = q_embs_list + pos_embs_list + neg_embs_list
    global_max_seq = max(t.size(1) for t in all_embs)

    def pad_to(t: torch.Tensor, target: int) -> torch.Tensor:
        diff = target - t.size(1)
        if diff <= 0:
            return t
        if t.dim() == 3:
            return F.pad(t, (0, 0, 0, diff))
        return F.pad(t, (0, diff))

    q_embs = torch.cat([pad_to(t, global_max_seq) for t in q_embs_list], dim=0)
    q_masks = torch.cat([pad_to(m, global_max_seq) for m in q_masks_list], dim=0)
    pos_embs = torch.cat([pad_to(t, global_max_seq) for t in pos_embs_list], dim=0)
    pos_masks = torch.cat([pad_to(m, global_max_seq) for m in pos_masks_list], dim=0)
    neg_embs = torch.cat([pad_to(t, global_max_seq) for t in neg_embs_list], dim=0)
    neg_masks = torch.cat([pad_to(m, global_max_seq) for m in neg_masks_list], dim=0)

    k_embs = torch.cat([pos_embs, neg_embs], dim=0)
    k_masks = torch.cat([pos_masks, neg_masks], dim=0)

    return q_embs, q_masks, k_embs, k_masks


def pad_and_cat_1d(tensors: List[torch.Tensor], target_seq: int) -> torch.Tensor:
    """Pad (B, S) ID tensors to target_seq and concatenate."""
    padded = [F.pad(t, (0, target_seq - t.size(1))) for t in tensors]
    return torch.cat(padded, dim=0)


def build_qrels(
    n_queries: int,
    n_corpus: int,
    target_indices: Optional[List[List[int]]] = None,
) -> tuple[Qrels, Qrels]:
    """Build hard and soft qrels for retrieval evaluation.

    Hard qrels use only the diagonal positive (index ``i`` for query ``i``).
    Soft qrels additionally include author-sharing positives from
    ``target_indices`` when available; otherwise soft falls back to hard.

    Parameters
    ----------
    n_queries : int
        Number of queries.
    n_corpus : int
        Total corpus size (positives + negatives).
    target_indices : list of list of int, optional
        For each query, a list of additional relevant document indices
        (from the same author set). ``None`` means use hard qrels only.

    Returns
    -------
    tuple[Qrels, Qrels]
        ``(hard_qrels, soft_qrels)``.
    """
    hard = {f"q{i}": {f"d{i}": 1} for i in range(n_queries)}

    if target_indices is not None:
        soft = {
            f"q{i}": {f"d{t}": 1 for t in target_indices[i] if 0 <= t < n_corpus}
            for i in range(n_queries)
        }
    else:
        soft = hard

    return Qrels(hard), Qrels(soft)


def scores_to_run(scores: torch.Tensor, k: int) -> Run:
    """Convert a dense similarity matrix into a top-k ranx Run object.

    Parameters
    ----------
    scores : torch.Tensor
        ``(n_queries, n_corpus)`` similarity matrix.
    k : int
        Number of documents to retrieve per query.

    Returns
    -------
    Run
        Ranx Run with top-k scored documents per query.
    """
    topk_scores, topk_indices = scores.topk(min(k, scores.size(1)), dim=1)
    return Run(
        {
            f"q{i}": {
                f"d{int(topk_indices[i, j])}": float(topk_scores[i, j])
                for j in range(topk_indices.size(1))
            }
            for i in range(scores.size(0))
        }
    )


def evaluate_run(
    hard_qrels: Qrels, soft_qrels: Qrels, run: Run, k: int
) -> Dict[str, float]:
    """Compute MRR, nDCG, Recall, and accuracy at depth k.

    Parameters
    ----------
    hard_qrels : Qrels
        Qrels with only the diagonal positive (used for MRR and accuracy).
    soft_qrels : Qrels
        Qrels with all author-sharing positives (used for nDCG and Recall).
    run : Run
        Ranked list of retrieved documents.
    k : int
        Cutoff depth.

    Returns
    -------
    dict
        Keys: ``mrr@k``, ``ndcg@k``, ``recall@k``, ``accuracy``.
    """
    return {
        f"mrr@{k}": evaluate(hard_qrels, run, f"mrr@{k}"),
        f"ndcg@{k}": evaluate(soft_qrels, run, f"ndcg@{k}"),
        f"recall@{k}": evaluate(soft_qrels, run, f"recall@{k}"),
        "accuracy": evaluate(hard_qrels, run, "precision@1"),
    }


def gather_targets(target_batches: List[torch.Tensor]) -> Optional[List[List[int]]]:
    """Flatten per-batch target-index tensors into a per-query list.

    Parameters
    ----------
    target_batches : list of Tensor
        Each tensor has shape ``(B, max_targets)`` with -1 as a sentinel for
        missing entries.

    Returns
    -------
    list of list of int, or None
        Per-query lists of valid (non-negative) target document indices.
        Returns ``None`` if no valid targets exist (triggers hard-qrel fallback).
    """
    if not target_batches:
        return None

    all_targets = []
    has_any_targets = False  # Track if we find at least one valid target

    for batch_t in target_batches:
        for row in batch_t:
            valid_targets = row[row >= 0].tolist()
            if valid_targets:
                has_any_targets = True
            all_targets.append(valid_targets)

    # If every single row was empty, return None to trigger the soft=hard fallback
    return all_targets if has_any_targets else None
