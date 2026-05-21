#!/usr/bin/env python3
# deep_stylometry/experiments/retrieval_inspection.py
"""Retrieval failure-mode analysis for the PLI / HALvest EMNLP paper.

Loads a trained checkpoint, builds a closed-set retrieval pool from the
validation split (deduplicated by ``pos_halid``), runs full ranking over
``n_seeds × n_queries_per_seed`` sampled queries, and emits failure-mode
statistics comparing high-ranked distractors to a random-pair baseline.

PAN19 is deferred: if the config specifies ``ds_name="pan19"`` a
``NotImplementedError`` is raised (different pool construction required).

Run::

    python -m deep_stylometry.experiments.retrieval_inspection \\
        --config_path $CONFIGS/test_pli_ngram3.yml \\
        --checkpoint_path tmp/modernbert__halvest__pli__ngram3/last.ckpt \\
        --subset base-4 \\
        --n_seeds 5 \\
        --n_queries_per_seed 100 \\
        --top_k 20 \\
        --output_dir ./analysis/retrieval_inspection_base4_ngram3 \\
        --cache_dir $HF_HOME \\
        --batch_size 32 \\
        --precision bf16-mixed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

from deep_stylometry.utils.retrieval_aggregation import (
    _aggregate_pair_records,
    _sanitize_for_json,
    _write_report,
)
from deep_stylometry.utils.text_stats import (
    _flatten_domain,
    _jaccard,
    _to_frozenset,
    punct_density_from_ids,
    sentence_length_std,
    span_token_count,
)

logger = logging.getLogger(__name__)

HALVEST_NAME = "almanach/halvest-contrastive"

# Number of pool documents scored per GPU call (avoids full (Nq, Np, q_len, k_len) tensor).
_DEFAULT_CHUNK_SIZE = 1024


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------


def _detect_columns(cols: List[str]) -> Dict[str, Optional[str]]:
    """Detect HALvest-Contrastive column names at runtime.

    Args:
        cols: Dataset column names.

    Returns:
        Dict mapping role → column name (or ``None`` if not found).
    """
    return {
        "query": next(
            (c for c in ("query", "anchor", "text", "sentence") if c in cols), None
        ),
        "positive": next(
            (c for c in ("positive", "pos_text", "document") if c in cols), None
        ),
        "pos_halid": next(
            (c for c in ("pos_halid", "halid", "pos_id") if c in cols), None
        ),
        "pos_authorids": next(
            (c for c in ("pos_authorids", "authorids", "author_ids") if c in cols), None
        ),
        "query_domain": next(
            (c for c in ("query_domain", "domain", "field") if c in cols), None
        ),
        "year": next((c for c in ("year", "pos_year", "pub_year") if c in cols), None),
    }


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------


def build_pool(
    ds: Any,
    col_map: Dict[str, Optional[str]],
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, int]]:
    """Build the deduplicated retrieval pool from a dataset split.

    The pool is keyed by ``pos_halid``: the first occurrence per halid is
    kept and ``positive`` text is used as the document body.  This is the
    single source of truth for locating each query's true positive — do NOT
    use validation-row indices as pool indices.

    Args:
        ds: HuggingFace Dataset (typically the validation split).
        col_map: Column-role mapping from :func:`_detect_columns`.

    Returns:
        ``(pool_texts, pool_meta, halid_to_idx)`` where:

        - ``pool_texts``: list of document text strings, length = pool size.
        - ``pool_meta``: list of metadata dicts
          ``{pos_halid, authorids, domain, domain_raw, year, row_idx}``.
        - ``halid_to_idx``: ``{pos_halid: pool_index}`` mapping.
    """
    halid_col = col_map["pos_halid"]
    text_col = col_map["positive"]
    auth_col = col_map["pos_authorids"]
    domain_col = col_map["query_domain"]
    year_col = col_map["year"]

    if halid_col is None:
        raise ValueError(
            f"Cannot locate pos_halid column in dataset. "
            f"Columns available: {ds.column_names}"
        )
    if text_col is None:
        raise ValueError(
            f"Cannot locate positive-text column. Columns: {ds.column_names}"
        )

    seen: Set[str] = set()
    pool_texts: List[str] = []
    pool_meta: List[Dict[str, Any]] = []
    halid_to_idx: Dict[str, int] = {}

    for i, row in enumerate(ds):
        halid = str(row[halid_col])
        if halid in seen:
            continue
        seen.add(halid)
        idx = len(pool_texts)
        halid_to_idx[halid] = idx

        raw_domain = row.get(domain_col) if domain_col else None
        pool_texts.append(row[text_col])
        pool_meta.append(
            {
                "pos_halid": halid,
                "authorids": _to_frozenset(row.get(auth_col) if auth_col else None),
                "domain": _flatten_domain(raw_domain),
                "domain_raw": raw_domain,
                "year": row.get(year_col) if year_col else None,
                "row_idx": i,
                "_pool_idx": idx,
            }
        )

    logger.info(
        "Pool: %d unique pos_halids from %d validation rows.", len(pool_texts), len(ds)
    )
    return pool_texts, pool_meta, halid_to_idx


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_texts(
    texts: List[str],
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a list of texts in batches with the encoder model.

    Args:
        texts: Text strings to encode.
        model: Encoder whose ``forward(input_ids, attention_mask)`` returns
               ``(batch, seq, hidden)`` token embeddings.
        tokenizer: HuggingFace tokenizer.
        device: Target device.
        dtype: Floating-point dtype for embeddings.
        batch_size: Texts per forward pass.
        max_length: Truncation/padding length.

    Returns:
        ``(all_embs, all_masks, all_ids)`` stored on ``device``:

        - ``all_embs``: ``(N, max_length, hidden)``
        - ``all_masks``: ``(N, max_length)`` long
        - ``all_ids``: ``(N, max_length)`` long
    """
    model.eval()
    all_embs: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []
    all_ids: List[torch.Tensor] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                embs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            embs = model(input_ids=input_ids, attention_mask=attention_mask)

        all_embs.append(embs.to(dtype))
        all_masks.append(attention_mask)
        all_ids.append(input_ids)

    return (
        torch.cat(all_embs, dim=0),
        torch.cat(all_masks, dim=0),
        torch.cat(all_ids, dim=0),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _call_scorer(
    scorer: nn.Module,
    q_embs: torch.Tensor,
    k_embs: torch.Tensor,
    q_mask: torch.Tensor,
    k_mask: torch.Tensor,
    q_ids: torch.Tensor,
    k_ids: torch.Tensor,
) -> torch.Tensor:
    """Dispatch a scoring call to the correct module, passing only the args
    each accepts.

    Args:
        scorer: ``LateInteraction``, ``MeanInteraction``, or ``PatchInteraction``.
        q_embs: ``(Bq, S, H)``
        k_embs: ``(Bk, S, H)``
        q_mask: ``(Bq, S)``
        k_mask: ``(Bk, S)``
        q_ids:  ``(Bq, S)``
        k_ids:  ``(Bk, S)``

    Returns:
        scores: ``(Bq, Bk)``
    """
    # DECISION: Use isinstance to dispatch rather than a registry or duck-typing
    # because the scoring modules are part of the deep_stylometry public API and
    # the arg differences are structural (LateInteraction takes q_input_ids but
    # not k_input_ids; PatchInteraction takes both; MeanInteraction takes neither).
    from deep_stylometry.modules.late_interaction import LateInteraction
    from deep_stylometry.modules.patch_interaction import PatchInteraction

    if isinstance(scorer, PatchInteraction):
        return scorer(q_embs, k_embs, q_mask, k_mask, q_ids, k_ids, step=None)
    if isinstance(scorer, LateInteraction):
        return scorer(q_embs, k_embs, q_mask, k_mask, q_ids)
    # MeanInteraction (and any future mean-like head): no input_ids needed.
    return scorer(q_embs, k_embs, q_mask, k_mask)


@torch.no_grad()
def score_query_vs_pool(
    q_embs: torch.Tensor,
    q_mask: torch.Tensor,
    q_ids: torch.Tensor,
    pool_embs: torch.Tensor,
    pool_masks: torch.Tensor,
    pool_ids: torch.Tensor,
    scorer: nn.Module,
    chunk_size: int,
) -> torch.Tensor:
    """Score one query against the entire pool using chunked iteration.

    Chunking is unconditional regardless of the scoring head so the script
    works on any pooling method without branching and without materialising a
    ``(1, N, q_len, k_len)`` tensor for late-interaction heads.

    Args:
        q_embs: ``(1, S, H)``
        q_mask: ``(1, S)``
        q_ids:  ``(1, S)``
        pool_embs: ``(N, S, H)`` on GPU.
        pool_masks: ``(N, S)`` on GPU.
        pool_ids:   ``(N, S)`` on GPU.
        scorer: Scoring module.
        chunk_size: Pool documents per scoring call.

    Returns:
        scores: ``(N,)`` on the same device as ``pool_embs``.
    """
    N = pool_embs.size(0)
    scores_parts: List[torch.Tensor] = []
    for j in range(0, N, chunk_size):
        chunk_scores = _call_scorer(
            scorer,
            q_embs,
            pool_embs[j : j + chunk_size],
            q_mask,
            pool_masks[j : j + chunk_size],
            q_ids,
            pool_ids[j : j + chunk_size],
        )
        scores_parts.append(chunk_scores.squeeze(0))
    return torch.cat(scores_parts, dim=0)


# ---------------------------------------------------------------------------
# Per-pair record construction
# ---------------------------------------------------------------------------


def _domain_set(raw: Any) -> FrozenSet[str]:
    """Convert a raw domain value to a frozenset (for Jaccard scoring)."""
    if isinstance(raw, list):
        return frozenset(str(d) for d in raw if d)
    if raw:
        return frozenset([str(raw)])
    return frozenset()


def _build_pair_record(
    rank: int,
    score: float,
    cand_pool_idx: int,
    pool_meta: List[Dict[str, Any]],
    query_meta: Dict[str, Any],
    query_text: str,
    pool_text_sets: Dict[int, FrozenSet[str]],
    pool_stats: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the full per-pair measurement record.

    Args:
        rank: 1-indexed candidate rank for this query.
        score: Raw similarity score.
        cand_pool_idx: Pool index of the candidate.
        pool_meta: Per-pool-document metadata list.
        query_meta: Query metadata dict (keys: authorids, domain, domain_raw,
                    year, _tp_pool_idx, pos_halid).
        query_text: Raw query text string.
        pool_text_sets: Pre-cached ``{pool_idx: frozenset_of_lowercased_words}``.
        pool_stats: Pre-cached ``{pool_idx: {tokens_post_trunc, fertility, punct_density}}``.

    Returns:
        Dict with all per-pair measurements listed in the spec.
    """
    cand = pool_meta[cand_pool_idx]
    tp_pool_idx = query_meta["_tp_pool_idx"]
    tp = pool_meta[tp_pool_idx]

    cand_auth: FrozenSet[str] = cand["authorids"]
    query_auth: FrozenSet[str] = query_meta["authorids"]
    tp_auth: FrozenSet[str] = tp["authorids"]

    cand_dom_set = _domain_set(cand["domain_raw"])
    tp_dom_set = _domain_set(tp["domain_raw"])

    cand_dom_str = cand["domain"]
    query_dom_str = query_meta["domain"]
    tp_dom_str = tp["domain"]

    # Year diffs (None if either year is missing).
    cy = cand.get("year")
    qy = query_meta.get("year")
    tpy = tp.get("year")
    year_diff_q = (int(cy) - int(qy)) if (cy is not None and qy is not None) else None
    year_diff_tp = (
        (int(cy) - int(tpy)) if (cy is not None and tpy is not None) else None
    )

    # Unigram Jaccard — use pre-populated pool_text_sets for pool docs.
    cand_uni = pool_text_sets[cand_pool_idx]
    tp_uni = pool_text_sets[tp_pool_idx]
    q_uni = frozenset(query_text.lower().split())

    # Pre-computed token stats for candidate.
    cstats = pool_stats.get(cand_pool_idx, {})

    return {
        "rank": rank,
        "score": score,
        "pool_idx": cand_pool_idx,
        "pos_halid": cand["pos_halid"],
        # author comparisons
        "author_jaccard_with_query": _jaccard(cand_auth, query_auth),
        "author_jaccard_with_true_positive": _jaccard(cand_auth, tp_auth),
        "exact_author_match_with_query": bool(cand_auth and cand_auth == query_auth),
        "exact_author_match_with_true_positive": bool(
            cand_auth and cand_auth == tp_auth
        ),
        "any_author_overlap_with_query": bool(cand_auth & query_auth),
        "any_author_overlap_with_true_positive": bool(cand_auth & tp_auth),
        # domain comparisons
        "same_domain_as_query": bool(cand_dom_str and cand_dom_str == query_dom_str),
        "same_domain_as_true_positive": bool(
            cand_dom_str and cand_dom_str == tp_dom_str
        ),
        "domain_jaccard_with_true_positive": _jaccard(cand_dom_set, tp_dom_set),
        # year diffs
        "year_diff_vs_query": year_diff_q,
        "year_diff_vs_true_positive": year_diff_tp,
        # unigram Jaccard (full text, not truncated)
        "unigram_jaccard_with_query": _jaccard(cand_uni, q_uni),
        "unigram_jaccard_with_true_positive": _jaccard(cand_uni, tp_uni),
        # candidate token stats (post-truncation)
        "tokens_post_trunc": cstats.get("tokens_post_trunc"),
        "fertility": cstats.get("fertility"),
        "punct_density": cstats.get("punct_density"),
        # metadata
        "candidate_authorids": sorted(str(a) for a in cand_auth),
        "candidate_domain": cand_dom_str,
        "candidate_year": cy,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    ds: Any,
    model: nn.Module,
    scorer: nn.Module,
    tokenizer: Any,
    output_dir: Path,
    subset: str,
    n_seeds: int,
    n_queries_per_seed: int,
    top_k: int,
    batch_size: int,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    max_length: int = 512,
    device: torch.device,
    dtype: torch.dtype,
    cfg_snapshot: Dict[str, Any],
    cli_args: Dict[str, Any],
    punct_set: Optional[Set[int]] = None,
    special_ids: Optional[Set[int]] = None,
) -> None:
    """Run the full retrieval inspection pipeline.

    This function is the testable entry point; ``main()`` loads the model and
    dataset and calls this.

    Args:
        ds: HuggingFace Dataset (validation split of the specified subset).
        model: Encoder module.
        scorer: Scoring module (LateInteraction / MeanInteraction / PatchInteraction).
        tokenizer: HuggingFace tokenizer.
        output_dir: Directory for all output files.
        subset: Subset name (informational).
        n_seeds: Number of query-sampling seeds.
        n_queries_per_seed: Queries to sample per seed.
        top_k: Candidates to rank per query.
        batch_size: Encoding batch size.
        chunk_size: Pool documents per scoring call.
        max_length: Token truncation/padding length.
        device: Compute device.
        dtype: Floating-point dtype for embeddings.
        cfg_snapshot: Effective config dict (written to config_snapshot.yml).
        cli_args: CLI arg dict (merged into config_snapshot.yml).
        punct_set: Optional punctuation token IDs for candidate stats.
        special_ids: Optional special token IDs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    col_map = _detect_columns(ds.column_names)
    query_col = col_map["query"]
    if query_col is None:
        raise ValueError(f"Cannot find query column in {ds.column_names}.")

    # ------------------------------------------------------------------ #
    # Build pool (once; identical across all seeds).                      #
    # ------------------------------------------------------------------ #
    logger.info("Building retrieval pool …")
    pool_texts, pool_meta, halid_to_idx = build_pool(ds, col_map)
    pool_size = len(pool_texts)

    # Pre-cache frozensets of lowercased word tokens for all pool docs
    # (used for unigram Jaccard without re-tokenising in the query loop).
    pool_text_sets: Dict[int, FrozenSet[str]] = {
        i: frozenset(t.lower().split()) for i, t in enumerate(pool_texts)
    }

    # Pre-compute token stats for all pool docs (post-truncation).
    pool_stats: Dict[int, Dict[str, Any]] = {}
    if punct_set is not None and special_ids is not None:
        logger.info("Pre-computing candidate token stats for %d pool docs …", pool_size)
        for i, text in enumerate(pool_texts):
            try:
                n_pre, n_post = span_token_count(text, tokenizer, max_length)
                from deep_stylometry.utils.text_stats import fertility as _fertility

                fert = _fertility(text, tokenizer)
                # Post-truncation token IDs via tokenizer call.
                enc = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=True,
                )
                pd = punct_density_from_ids(enc["input_ids"], punct_set, special_ids)
                pool_stats[i] = {
                    "tokens_post_trunc": n_post,
                    "fertility": fert,
                    "punct_density": pd,
                }
            except Exception:
                pool_stats[i] = {}

    # ------------------------------------------------------------------ #
    # Encode pool (once; held on GPU for the entire run).                 #
    # ------------------------------------------------------------------ #
    logger.info("Encoding %d pool documents …", pool_size)
    pool_embs, pool_masks, pool_ids = encode_texts(
        pool_texts, model, tokenizer, device, dtype, batch_size, max_length
    )
    logger.info(
        "Pool encoded: embs=%s masks=%s ids=%s (device=%s, dtype=%s)",
        pool_embs.shape,
        pool_masks.shape,
        pool_ids.shape,
        device,
        dtype,
    )

    # Save pool construction for post-hoc index-to-halid mapping.
    pool_halids_path = output_dir / "pool_halids.json"
    with open(pool_halids_path, "w", encoding="utf-8") as fh:
        json.dump(
            [m["pos_halid"] for m in pool_meta],
            fh,
            indent=2,
            sort_keys=True,
            default=str,
        )
    logger.info("Wrote %s", pool_halids_path)

    # ------------------------------------------------------------------ #
    # Per-seed query sampling and ranking.                                #
    # ------------------------------------------------------------------ #
    halid_col = col_map["pos_halid"]
    auth_col = col_map["pos_authorids"]
    domain_col = col_map["query_domain"]
    year_col = col_map["year"]

    all_pair_records: List[Dict[str, Any]] = []
    n_ds = len(ds)
    first_query_checked = False  # pool/query alignment sanity check (first query only)

    for seed_idx in range(n_seeds):
        rng = np.random.default_rng(seed_idx)
        query_row_indices = rng.choice(
            n_ds,
            size=min(n_queries_per_seed, n_ds),
            replace=False,
        ).tolist()

        # Also seed Python + torch for reproducibility within this seed.
        random.seed(seed_idx)
        torch.manual_seed(seed_idx)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed_idx)

        logger.info(
            "Seed %d/%d: %d queries …", seed_idx + 1, n_seeds, len(query_row_indices)
        )

        for q_row_idx in query_row_indices:
            row = ds[q_row_idx]
            q_text = row[query_col]

            q_halid = str(row[halid_col]) if halid_col else ""
            tp_pool_idx = halid_to_idx.get(q_halid)
            if tp_pool_idx is None:
                logger.debug(
                    "Query row %d halid=%r not in pool; skipping.", q_row_idx, q_halid
                )
                continue

            q_meta: Dict[str, Any] = {
                "pos_halid": q_halid,
                "authorids": _to_frozenset(row.get(auth_col) if auth_col else None),
                "domain": _flatten_domain(row.get(domain_col) if domain_col else None),
                "domain_raw": row.get(domain_col) if domain_col else None,
                "year": row.get(year_col) if year_col else None,
                "_tp_pool_idx": tp_pool_idx,
            }

            # Pool/query alignment sanity check on the first valid query.
            if not first_query_checked:
                first_query_checked = True
                if pool_meta[tp_pool_idx]["pos_halid"] != q_meta["pos_halid"]:
                    logger.error(
                        "ALIGNMENT MISMATCH on first query: "
                        "pool_meta[%d]['pos_halid']=%r != query pos_halid=%r. "
                        "Pool or halid_to_idx mapping may be broken.",
                        tp_pool_idx,
                        pool_meta[tp_pool_idx]["pos_halid"],
                        q_meta["pos_halid"],
                    )

            # Encode query (single example).
            q_enc = tokenizer(
                [q_text],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            q_input_ids = q_enc["input_ids"].to(device)
            q_attention_mask = q_enc["attention_mask"].to(device)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    q_embs = model(
                        input_ids=q_input_ids, attention_mask=q_attention_mask
                    ).to(dtype)
            else:
                q_embs = model(input_ids=q_input_ids, attention_mask=q_attention_mask)

            # Score against pool in chunks.
            scores = score_query_vs_pool(
                q_embs,
                q_attention_mask,
                q_input_ids,
                pool_embs,
                pool_masks,
                pool_ids,
                scorer,
                chunk_size,
            )

            # Full-pool rank of the true positive (1-indexed, always defined).
            # Midrank convention for ties: if k docs are tied with the TP,
            # the TP's rank is the average of its possible positions.
            tp_score = scores[tp_pool_idx].item()
            n_strictly_better = int((scores > tp_score).sum().item())
            n_ties = int((scores == tp_score).sum().item()) - 1  # exclude TP itself
            tp_rank: int = n_strictly_better + 1 + n_ties // 2

            # Top-K ranking (unchanged; builds per-pair candidate records).
            k_actual = min(top_k, pool_size)
            topk_scores, topk_indices = torch.topk(scores, k=k_actual, largest=True)
            topk_scores = topk_scores.cpu().float().tolist()
            topk_indices = topk_indices.cpu().tolist()

            # Build per-pair records for top-K candidates.
            for rank_0, (pool_idx, score_val) in enumerate(
                zip(topk_indices, topk_scores)
            ):
                rec = _build_pair_record(
                    rank=rank_0 + 1,
                    score=score_val,
                    cand_pool_idx=pool_idx,
                    pool_meta=pool_meta,
                    query_meta=q_meta,
                    query_text=q_text,
                    pool_text_sets=pool_text_sets,
                    pool_stats=pool_stats,
                )
                rec["seed"] = seed_idx
                rec["query_ds_idx"] = q_row_idx
                rec["true_positive_rank"] = tp_rank
                all_pair_records.append(rec)

    logger.info("Total pair records: %d", len(all_pair_records))

    # ------------------------------------------------------------------ #
    # Random baseline: n_seeds × n_queries random (query, candidate) pairs.
    # ------------------------------------------------------------------ #
    rng_baseline = np.random.default_rng(999)
    random_pairs: List[Dict[str, Any]] = []
    query_row_sample = rng_baseline.choice(
        n_ds, size=n_seeds * n_queries_per_seed, replace=True
    ).tolist()
    for q_row_idx in query_row_sample:
        row = ds[q_row_idx]
        q_halid = str(row[halid_col]) if halid_col else ""
        tp_pool_idx = halid_to_idx.get(q_halid)
        if tp_pool_idx is None:
            continue
        q_meta = {
            "pos_halid": q_halid,
            "authorids": _to_frozenset(row.get(auth_col) if auth_col else None),
            "domain": _flatten_domain(row.get(domain_col) if domain_col else None),
            "domain_raw": row.get(domain_col) if domain_col else None,
            "year": row.get(year_col) if year_col else None,
            "_tp_pool_idx": tp_pool_idx,
        }
        q_text = row[query_col]
        # Draw random candidate != true positive.
        for _ in range(10):
            cand_idx = int(rng_baseline.integers(0, pool_size))
            if cand_idx != tp_pool_idx:
                break
        rec = _build_pair_record(
            rank=0,  # not a ranked result
            score=float("nan"),
            cand_pool_idx=cand_idx,
            pool_meta=pool_meta,
            query_meta=q_meta,
            query_text=q_text,
            pool_text_sets=pool_text_sets,
            pool_stats=pool_stats,
        )
        rec["seed"] = -1
        rec["query_ds_idx"] = q_row_idx
        rec["true_positive_rank"] = None
        random_pairs.append(rec)

    # ------------------------------------------------------------------ #
    # Aggregate and write outputs.                                        #
    # ------------------------------------------------------------------ #
    summary = _aggregate_pair_records(all_pair_records, random_pairs, top_k)

    # pairs.jsonl
    pairs_path = output_dir / "pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as fh:
        for rec in all_pair_records:
            fh.write(json.dumps(rec, default=str) + "\n")
    logger.info("Wrote %d lines → %s", len(all_pair_records), pairs_path)

    # random_pairs.jsonl — persisted so retrieval_inspection_reaggregate.py
    # can recompute the ratio blocks without rerunning the GPU job.
    random_pairs_path = output_dir / "random_pairs.jsonl"
    with open(random_pairs_path, "w", encoding="utf-8") as fh:
        for rec in random_pairs:
            fh.write(json.dumps(rec, default=str) + "\n")
    logger.info("Wrote %d lines → %s", len(random_pairs), random_pairs_path)

    # summary.json — sanitize non-finite floats (e.g. inf ratios) to strings
    # before serialisation because float("inf") is not valid RFC-8259 JSON.
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(_sanitize_for_json(summary), fh, indent=2, sort_keys=True, default=str)
    logger.info("Wrote %s", summary_path)

    # report.txt
    report_path = output_dir / "report.txt"
    _write_report(report_path, summary, subset, n_seeds, n_queries_per_seed)

    rank1_pct = summary.get("overall_metrics", {}).get("rank1_accuracy_pct", 0.0)
    logger.info(
        "Rank-1 accuracy: %.1f%% — verify against test.py output for the same "
        "checkpoint to detect pool/checkpoint mismatch.",
        rank1_pct,
    )

    # config_snapshot.yml
    snapshot = {**cfg_snapshot, "_cli_args": cli_args}
    snap_path = output_dir / "config_snapshot.yml"
    with open(snap_path, "w", encoding="utf-8") as fh:
        yaml.dump(snapshot, fh, default_flow_style=False, indent=2)
    logger.info("Wrote %s", snap_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Retrieval failure-mode analysis: encode validation split, "
            "rank against deduplicated pool, emit failure statistics."
        )
    )
    p.add_argument(
        "--config_path",
        required=True,
        help="Path to test YAML config (e.g. configs/test_pli_ngram3.yml).",
    )
    p.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to Lightning checkpoint (.ckpt).",
    )
    p.add_argument(
        "--subset",
        default="base-4",
        help="HALvest-Contrastive subset (default: base-4).",
    )
    p.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of query-sampling seeds (default: 5).",
    )
    p.add_argument(
        "--n_queries_per_seed",
        type=int,
        default=1000,
        help="Queries to sample per seed (default: 100).",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Candidates to rank per query (default: 50).",
    )
    p.add_argument(
        "--output_dir",
        default="./analysis/retrieval_inspection",
        help="Directory for output files.",
    )
    p.add_argument(
        "--cache_dir",
        default=os.environ.get("HF_HOME"),
        help="HuggingFace cache directory.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Encoding batch size (default: 32).",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=_DEFAULT_CHUNK_SIZE,
        help=f"Pool docs per scoring call (default: {_DEFAULT_CHUNK_SIZE}).",
    )
    p.add_argument(
        "--precision",
        default="bf16-mixed",
        choices=["bf16-mixed", "16-mixed", "32"],
        help="Mixed-precision mode (default: bf16-mixed).",
    )
    return p


def main() -> None:
    """Load checkpoint and dataset, then run the retrieval inspection
    pipeline."""
    import datasets as hf_datasets

    from deep_stylometry.modules import DeepStylometry
    from deep_stylometry.utils.configs import BaseConfig
    from deep_stylometry.utils.helpers import get_tokenizer, resolve_lightning_precision
    from deep_stylometry.utils.text_stats import build_punct_token_id_set

    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = BaseConfig(mode="test").from_yaml(args.config_path)

    if cfg.data.ds_name == "pan19":
        raise NotImplementedError(
            "PAN19 failure-mode analysis requires a per-problem pool construction "
            "(9 candidates per problem, not a global pool) and a different notion "
            "of 'wrong high-ranked'. This will be added in a future PR. "
            "For HALvest subsets, pass a halvest config."
        )

    _, dtype = resolve_lightning_precision(args.precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model from %s …", args.checkpoint_path)
    model = DeepStylometry.load_from_checkpoint(args.checkpoint_path, cfg=cfg)
    model.eval()
    model.to(device)

    scorer = model.contrastive_loss.pool

    tokenizer = get_tokenizer(cfg.data.tokenizer_name)

    # Build punctuation set for candidate stats (optional; non-blocking on failure).
    punct_set: Optional[Set[int]] = None
    special_ids: Optional[Set[int]] = None
    try:
        punct_set = build_punct_token_id_set(cfg.data.tokenizer_name)
        special_ids = set()
        for attr in (
            "cls_token_id",
            "sep_token_id",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "mask_token_id",
        ):
            val = getattr(tokenizer, attr, None)
            if val is not None:
                special_ids.add(val)
    except Exception as exc:
        logger.warning("Could not build punct_set (%s); skipping token stats.", exc)

    logger.info("Loading valid split for subset '%s' …", args.subset)
    ds = hf_datasets.load_dataset(
        HALVEST_NAME,
        name=args.subset,
        split="valid",
        cache_dir=args.cache_dir,
    )
    logger.info("Valid split: %d rows, columns: %s", len(ds), ds.column_names)

    run_pipeline(
        ds=ds,
        model=model,
        scorer=scorer,
        tokenizer=tokenizer,
        output_dir=Path(args.output_dir),
        subset=args.subset,
        n_seeds=args.n_seeds,
        n_queries_per_seed=args.n_queries_per_seed,
        top_k=args.top_k,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        max_length=cfg.data.max_length,
        device=device,
        dtype=dtype,
        cfg_snapshot=cfg.to_dict(),
        cli_args=vars(args),
        punct_set=punct_set,
        special_ids=special_ids,
    )


if __name__ == "__main__":
    main()
