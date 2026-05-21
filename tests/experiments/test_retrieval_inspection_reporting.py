# tests/experiments/test_retrieval_inspection_reporting.py
"""Tests for the corrected ratio-block and full-pool ranking logic.

All tests run on CPU with no GPU, no checkpoint loading, and no real HF
dataset. Synthetic pair records are constructed in-memory.

Run::

    python -m pytest tests/experiments/test_retrieval_inspection_reporting.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair_record(
    seed: int = 0,
    query_ds_idx: int = 0,
    tp_rank: Optional[int] = 1,
    author_jaccard_with_query: float = 0.0,
    same_domain_as_query: float = 0.0,
    unigram_jaccard_with_query: float = 0.0,
) -> Dict[str, Any]:
    """Construct a minimal pair record with all fields _bucket_stats reads."""
    return {
        "seed": seed,
        "query_ds_idx": query_ds_idx,
        "true_positive_rank": tp_rank,
        "rank": 1,
        "score": 0.9,
        "pool_idx": 0,
        "pos_halid": f"h{query_ds_idx}",
        "author_jaccard_with_query": author_jaccard_with_query,
        "author_jaccard_with_true_positive": 0.0,
        "same_domain_as_query": same_domain_as_query,
        "same_domain_as_true_positive": 0.0,
        "domain_jaccard_with_true_positive": 0.0,
        "unigram_jaccard_with_query": unigram_jaccard_with_query,
        "unigram_jaccard_with_true_positive": 0.0,
        "any_author_overlap_with_query": False,
        "any_author_overlap_with_true_positive": False,
    }


def _make_distractor_record(
    query_ds_idx: int = 0,
    author_jaccard_with_query: float = 0.0,
    same_domain_as_query: float = 0.0,
    unigram_jaccard_with_query: float = 0.0,
) -> Dict[str, Any]:
    """Pair record with tp_rank=2 so it lands in the rank=2-5 distractor bucket."""
    return _make_pair_record(
        seed=0,
        query_ds_idx=query_ds_idx,
        tp_rank=2,
        author_jaccard_with_query=author_jaccard_with_query,
        same_domain_as_query=same_domain_as_query,
        unigram_jaccard_with_query=unigram_jaccard_with_query,
    )


def _make_random_record(
    query_ds_idx: int = 100,
    author_jaccard_with_query: float = 0.0,
    same_domain_as_query: float = 0.0,
    unigram_jaccard_with_query: float = 0.0,
) -> Dict[str, Any]:
    """Random-pair record (seed=-1, tp_rank=None) for the baseline bucket."""
    rec = _make_pair_record(
        seed=-1,
        query_ds_idx=query_ds_idx,
        tp_rank=None,
        author_jaccard_with_query=author_jaccard_with_query,
        same_domain_as_query=same_domain_as_query,
        unigram_jaccard_with_query=unigram_jaccard_with_query,
    )
    rec["true_positive_rank"] = None  # explicit None for random pairs
    return rec


# ---------------------------------------------------------------------------
# Test 1: zero-baseline → ratio is inf, not None
# ---------------------------------------------------------------------------


def test_zero_baseline_ratio_is_inf_not_none() -> None:
    """When distractors have positive author-Jaccard but random baseline is 0,
    the ratio must be float('inf') with an explanatory note."""
    from deep_stylometry.utils.retrieval_aggregation import _aggregate_pair_records

    pair_records = [_make_distractor_record(author_jaccard_with_query=0.1)]
    random_pairs = [
        _make_random_record(author_jaccard_with_query=0.0, query_ds_idx=i)
        for i in range(100, 110)
    ]

    summary = _aggregate_pair_records(pair_records, random_pairs, top_k=5)

    block = summary["headline_ratios"]["author_jaccard_with_query"]
    assert block["ratio"] == float("inf"), (
        f"Expected float('inf'), got {block['ratio']!r}"
    )
    assert block["note"] is not None
    assert "essentially never present under chance pairing" in block["note"], (
        f"Note text unexpected: {block['note']!r}"
    )
    assert block["distractor_mean"] == pytest.approx(0.1)
    assert block["random_mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 2: both-zero baseline → ratio is None with explanatory note
# ---------------------------------------------------------------------------


def test_both_zero_baseline_ratio_is_none_with_note() -> None:
    """When both distractor mean and random mean are 0.0, ratio=None and
    the note explains why."""
    from deep_stylometry.utils.retrieval_aggregation import _aggregate_pair_records

    pair_records = [_make_distractor_record(author_jaccard_with_query=0.0)]
    random_pairs = [
        _make_random_record(author_jaccard_with_query=0.0, query_ds_idx=i)
        for i in range(100, 110)
    ]

    summary = _aggregate_pair_records(pair_records, random_pairs, top_k=5)

    block = summary["headline_ratios"]["author_jaccard_with_query"]
    assert block["ratio"] is None, f"Expected None, got {block['ratio']!r}"
    assert block["note"] is not None
    assert "Both distractor and random baselines are zero" in block["note"], (
        f"Note text unexpected: {block['note']!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: normal nonzero ratio computed correctly
# ---------------------------------------------------------------------------


def test_normal_ratio_computed_correctly() -> None:
    """With distractor_mean=0.2 and random_mean=0.05, ratio must equal 4.0."""
    from deep_stylometry.utils.retrieval_aggregation import _aggregate_pair_records

    pair_records = [_make_distractor_record(author_jaccard_with_query=0.2)]
    random_pairs = [
        _make_random_record(author_jaccard_with_query=0.05, query_ds_idx=i)
        for i in range(100, 110)
    ]

    summary = _aggregate_pair_records(pair_records, random_pairs, top_k=5)

    block = summary["headline_ratios"]["author_jaccard_with_query"]
    assert block["ratio"] == pytest.approx(4.0), (
        f"Expected ratio=4.0, got {block['ratio']!r}"
    )
    assert block["note"] is None


# ---------------------------------------------------------------------------
# Test 4: full-pool ranking — no ties
# ---------------------------------------------------------------------------


def test_full_pool_ranking_no_ties() -> None:
    """With a clear score ordering, global TP rank equals 1 + count of strictly
    higher scores."""
    scores = torch.tensor([0.5, 0.9, 0.3, 0.7])
    tp_pool_idx = 3  # score 0.7; one doc (idx 1, score 0.9) is strictly better

    tp_score = scores[tp_pool_idx].item()
    n_strictly_better = int((scores > tp_score).sum().item())
    n_ties = int((scores == tp_score).sum().item()) - 1
    tp_rank = n_strictly_better + 1 + n_ties // 2

    assert tp_rank == 2, f"Expected rank 2, got {tp_rank}"


def test_full_pool_ranking_tp_is_best() -> None:
    """True positive has the highest score → rank 1."""
    scores = torch.tensor([0.9, 0.3, 0.5, 0.7])
    tp_pool_idx = 0  # score 0.9; nothing strictly better

    tp_score = scores[tp_pool_idx].item()
    n_strictly_better = int((scores > tp_score).sum().item())
    n_ties = int((scores == tp_score).sum().item()) - 1
    tp_rank = n_strictly_better + 1 + n_ties // 2

    assert tp_rank == 1, f"Expected rank 1, got {tp_rank}"


# ---------------------------------------------------------------------------
# Test 5: full-pool ranking — ties use midrank
# ---------------------------------------------------------------------------


def test_full_pool_ranking_ties_use_midrank() -> None:
    """Three documents tied at the highest score, one is the TP.
    Midrank = 0 strictly_better + 1 + (2 ties)//2 = 2."""
    scores = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.3])
    tp_pool_idx = 1  # tied at score 1.0 with indices 0 and 2

    tp_score = scores[tp_pool_idx].item()
    n_strictly_better = int((scores > tp_score).sum().item())
    n_ties = int((scores == tp_score).sum().item()) - 1  # 2 other tied docs
    tp_rank = n_strictly_better + 1 + n_ties // 2

    assert n_strictly_better == 0
    assert n_ties == 2
    assert tp_rank == 2, (
        f"Expected midrank 2 for 3-way tie at top, got {tp_rank}"
    )


# ---------------------------------------------------------------------------
# Test 6: reaggregate produces same summary as direct call
# ---------------------------------------------------------------------------


def test_reaggregate_matches_direct_aggregation(tmp_path: Path) -> None:
    """Writing pairs.jsonl and random_pairs.jsonl then running reaggregate
    must produce the same summary as calling _aggregate_pair_records directly."""
    from deep_stylometry.experiments.retrieval_inspection_reaggregate import reaggregate
    from deep_stylometry.utils.retrieval_aggregation import _aggregate_pair_records

    pair_records = [
        _make_distractor_record(query_ds_idx=i, author_jaccard_with_query=0.2)
        for i in range(5)
    ]
    random_pairs = [
        _make_random_record(query_ds_idx=100 + i, author_jaccard_with_query=0.05)
        for i in range(10)
    ]
    top_k = 5

    # Write JSONL files.
    (tmp_path / "pairs.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in pair_records) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "random_pairs.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in random_pairs) + "\n",
        encoding="utf-8",
    )

    # Re-aggregate from files (stale_ranks=True because pairs have tp_rank=2,
    # which is a valid integer — no None ranks here, so the warning is NOT
    # triggered; we just use the stale_ranks path for the reaggregate tool).
    reaggregated = reaggregate(
        input_dir=tmp_path,
        output_dir=tmp_path,
        top_k=top_k,
    )

    # Direct aggregation (stale_ranks=False since ranks are valid integers).
    direct = _aggregate_pair_records(pair_records, random_pairs, top_k)

    # The ratio values and distractor/random means must match.
    for key in ("author_jaccard_with_query", "same_domain_as_query", "unigram_jaccard_with_query"):
        ra_block = reaggregated["headline_ratios"][key]
        di_block = direct["headline_ratios"][key]
        # Both None is fine; if one is numeric the other must agree.
        if di_block["ratio"] is None:
            assert ra_block["ratio"] is None, (
                f"Expected None ratio for {key}, got {ra_block['ratio']!r}"
            )
        else:
            assert ra_block["ratio"] == pytest.approx(di_block["ratio"], abs=1e-6), (
                f"Ratio mismatch for {key}: reaggregated={ra_block['ratio']!r}, "
                f"direct={di_block['ratio']!r}"
            )
        if di_block["distractor_mean"] is None:
            assert ra_block["distractor_mean"] is None
        else:
            assert ra_block["distractor_mean"] == pytest.approx(
                di_block["distractor_mean"], abs=1e-9
            )

    # Overall metrics must match.
    assert reaggregated["overall_metrics"]["rank1_accuracy_pct"] == pytest.approx(
        direct["overall_metrics"]["rank1_accuracy_pct"], abs=0.01
    )


# ---------------------------------------------------------------------------
# Test 7: reaggregate handles stale tp_rank=None with warning + >top_k bucket
# ---------------------------------------------------------------------------


def test_reaggregate_stale_ranks_warning_and_bucket(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """pairs.jsonl with all tp_rank=None must:
    1. Run to completion (no exception).
    2. Place all queries in the '>top_k' bucket.
    3. Print a warning to stderr.
    """
    from deep_stylometry.experiments.retrieval_inspection_reaggregate import reaggregate

    # All pair records have tp_rank=None (old-style pairs.jsonl).
    pair_records = [
        _make_random_record(
            query_ds_idx=i, author_jaccard_with_query=0.1
        )
        for i in range(5)
    ]
    # Give each a distinct (seed, query_ds_idx) so they are counted as 5 queries.
    for i, rec in enumerate(pair_records):
        rec["seed"] = i
        rec["query_ds_idx"] = i

    random_pairs = [
        _make_random_record(query_ds_idx=100 + i, author_jaccard_with_query=0.0)
        for i in range(10)
    ]

    (tmp_path / "pairs.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in pair_records) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "random_pairs.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in random_pairs) + "\n",
        encoding="utf-8",
    )

    summary = reaggregate(
        input_dir=tmp_path,
        output_dir=tmp_path,
        top_k=20,
    )

    # All 5 queries must land in ">top_k".
    counts = summary["rank_distribution"]["counts"]
    assert counts.get(">top_k", 0) == 5, (
        f"Expected 5 queries in '>top_k', got counts={counts}"
    )
    # All other rank bins must be empty.
    for b in ["1", "2-5", "6-10", "11-20", "21-50", "51-100", "101-500", ">500"]:
        assert counts.get(b, 0) == 0, (
            f"Bin '{b}' should be empty for stale data, got {counts.get(b)}"
        )

    # Warning must appear on stderr.
    captured = capsys.readouterr()
    assert "WARNING" in captured.err or "cannot be recovered" in captured.err, (
        f"Expected a warning on stderr, got: {captured.err!r}"
    )
