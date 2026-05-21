# tests/experiments/test_retrieval_inspection.py
"""Unit and smoke tests for the new retrieval_inspection pipeline.

Three test classes:

1. ``TestPoolConstruction`` — pool built from a tiny mock dataset; deduplication
   and halid_to_idx consistency verified.
2. ``TestPairRecord`` — per-pair measurement function tested against
   hand-constructed metadata; novel ``author_jaccard_with_true_positive`` field
   is the primary focus.
3. ``TestSmokePipeline`` — end-to-end smoke test with a mock encoder and
   MeanInteraction scorer, no checkpoint, no HF download, no GPU required.

Run::

    python -m pytest tests/experiments/test_retrieval_inspection.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Class 1: Pool construction
# ---------------------------------------------------------------------------

class TestPoolConstruction:
    """build_pool() must deduplicate by pos_halid and keep consistent mapping."""

    def _make_mock_ds(self):
        """Build a 10-row in-memory dataset with 4 unique pos_halids."""
        import datasets as hf_datasets  # in-memory only, no download

        halids = ["h1", "h2", "h1", "h3", "h2", "h4", "h3", "h1", "h4", "h2"]
        texts = [f"Document text for halid {h}, row {i}" for i, h in enumerate(halids)]
        author_lists = [
            ["a1", "a2"], ["a3"], ["a1", "a2"], ["a4"],
            ["a3"], ["a5"], ["a4"], ["a1", "a2"], ["a5"], ["a3"],
        ]
        domains = ["cs"] * 5 + ["phys"] * 5

        return hf_datasets.Dataset.from_dict(
            {
                "pos_halid": halids,
                "positive": texts,
                "query": [f"Query {i}" for i in range(10)],
                "pos_authorids": author_lists,
                "query_domain": domains,
            }
        )

    def test_deduplication_count(self) -> None:
        """Pool should contain exactly 4 unique halids from 10 rows."""
        from deep_stylometry.experiments.retrieval_inspection import (
            _detect_columns,
            build_pool,
        )

        ds = self._make_mock_ds()
        col_map = _detect_columns(ds.column_names)
        pool_texts, pool_meta, halid_to_idx = build_pool(ds, col_map)

        assert len(pool_texts) == 4, f"Expected 4 unique docs, got {len(pool_texts)}"
        assert len(pool_meta) == 4
        assert len(halid_to_idx) == 4

    def test_first_occurrence_kept(self) -> None:
        """First occurrence of each halid is the one kept in the pool."""
        from deep_stylometry.experiments.retrieval_inspection import (
            _detect_columns,
            build_pool,
        )

        ds = self._make_mock_ds()
        col_map = _detect_columns(ds.column_names)
        pool_texts, pool_meta, halid_to_idx = build_pool(ds, col_map)

        # h1 first appears at row 0 with text "Document text for halid h1, row 0"
        h1_idx = halid_to_idx["h1"]
        assert "row 0" in pool_texts[h1_idx], (
            f"Expected first occurrence text, got: {pool_texts[h1_idx]}"
        )

    def test_halid_to_idx_consistency(self) -> None:
        """halid_to_idx[halid] must equal pool_meta[idx]['_pool_idx']."""
        from deep_stylometry.experiments.retrieval_inspection import (
            _detect_columns,
            build_pool,
        )

        ds = self._make_mock_ds()
        col_map = _detect_columns(ds.column_names)
        _, pool_meta, halid_to_idx = build_pool(ds, col_map)

        for halid, idx in halid_to_idx.items():
            assert pool_meta[idx]["pos_halid"] == halid, (
                f"Mismatch: halid_to_idx[{halid!r}]={idx}, "
                f"pool_meta[{idx}]['pos_halid']={pool_meta[idx]['pos_halid']!r}"
            )
            assert pool_meta[idx]["_pool_idx"] == idx

    def test_all_halids_covered(self) -> None:
        """Every unique halid in the dataset must appear in halid_to_idx."""
        from deep_stylometry.experiments.retrieval_inspection import (
            _detect_columns,
            build_pool,
        )

        ds = self._make_mock_ds()
        col_map = _detect_columns(ds.column_names)
        _, _, halid_to_idx = build_pool(ds, col_map)

        expected = {"h1", "h2", "h3", "h4"}
        assert set(halid_to_idx.keys()) == expected

    def test_missing_halid_column_raises(self) -> None:
        """Raising ValueError when pos_halid column is absent."""
        import datasets as hf_datasets
        from deep_stylometry.experiments.retrieval_inspection import (
            _detect_columns,
            build_pool,
        )

        ds = hf_datasets.Dataset.from_dict(
            {"positive": ["doc1", "doc2"], "query": ["q1", "q2"]}
        )
        col_map = _detect_columns(ds.column_names)
        with pytest.raises(ValueError, match="pos_halid"):
            build_pool(ds, col_map)


# ---------------------------------------------------------------------------
# Class 2: Per-pair record measurements
# ---------------------------------------------------------------------------

class TestPairRecord:
    """_build_pair_record() must compute all measurement fields correctly."""

    def _make_pool_meta(self) -> List[Dict[str, Any]]:
        return [
            {
                "pos_halid": "h1",
                "authorids": frozenset({"a1", "a2"}),
                "domain": "cs",
                "domain_raw": "cs",
                "year": 2020,
                "_pool_idx": 0,
            },
            {
                "pos_halid": "h2",
                "authorids": frozenset({"a3"}),
                "domain": "phys",
                "domain_raw": "phys",
                "year": 2021,
                "_pool_idx": 1,
            },
            {
                "pos_halid": "h3",
                "authorids": frozenset({"a1", "a4"}),
                "domain": "cs",
                "domain_raw": "cs",
                "year": 2019,
                "_pool_idx": 2,
            },
        ]

    def _make_query_meta(self, tp_pool_idx: int) -> Dict[str, Any]:
        return {
            "pos_halid": "h1",
            "authorids": frozenset({"a1", "a2"}),
            "domain": "cs",
            "domain_raw": "cs",
            "year": 2020,
            "_tp_pool_idx": tp_pool_idx,
        }

    def test_author_jaccard_with_true_positive_at_rank1(self) -> None:
        """Rank-1 candidate IS the true positive → author_jaccard_with_tp = 1.0."""
        from deep_stylometry.experiments.retrieval_inspection import _build_pair_record

        pool_meta = self._make_pool_meta()
        q_meta = self._make_query_meta(tp_pool_idx=0)

        pool_text_sets = {
            0: frozenset(["document", "one"]),
            1: frozenset(["document", "two"]),
            2: frozenset(["document", "three"]),
        }
        pool_stats: Dict[int, Dict[str, Any]] = {}

        rec = _build_pair_record(
            rank=1,
            score=0.95,
            cand_pool_idx=0,  # same as tp
            pool_meta=pool_meta,
            query_meta=q_meta,
            query_text="document one query",
            pool_text_sets=pool_text_sets,
            pool_stats=pool_stats,
        )

        assert rec["author_jaccard_with_true_positive"] == pytest.approx(1.0), (
            "Rank-1 candidate = true positive; author Jaccard with TP must be 1.0"
        )
        assert rec["exact_author_match_with_true_positive"] is True
        assert rec["any_author_overlap_with_true_positive"] is True

    def test_author_jaccard_with_true_positive_partial_overlap(self) -> None:
        """Candidate shares one author with TP but not all → 0 < jaccard < 1."""
        from deep_stylometry.experiments.retrieval_inspection import _build_pair_record

        pool_meta = self._make_pool_meta()
        q_meta = self._make_query_meta(tp_pool_idx=0)

        pool_text_sets = {i: frozenset() for i in range(3)}
        pool_stats = {}

        rec = _build_pair_record(
            rank=2,
            score=0.8,
            cand_pool_idx=2,  # authorids = {"a1", "a4"}; tp has {"a1", "a2"}
            pool_meta=pool_meta,
            query_meta=q_meta,
            query_text="some query text",
            pool_text_sets=pool_text_sets,
            pool_stats=pool_stats,
        )

        # Jaccard({a1, a4}, {a1, a2}) = 1/3
        assert rec["author_jaccard_with_true_positive"] == pytest.approx(1 / 3), (
            f"Expected 1/3, got {rec['author_jaccard_with_true_positive']}"
        )
        assert rec["any_author_overlap_with_true_positive"] is True
        assert rec["exact_author_match_with_true_positive"] is False

    def test_author_jaccard_with_true_positive_no_overlap(self) -> None:
        """Candidate has no author overlap with TP → jaccard = 0.0."""
        from deep_stylometry.experiments.retrieval_inspection import _build_pair_record

        pool_meta = self._make_pool_meta()
        q_meta = self._make_query_meta(tp_pool_idx=0)
        pool_text_sets = {i: frozenset() for i in range(3)}

        rec = _build_pair_record(
            rank=3,
            score=0.7,
            cand_pool_idx=1,  # authorids = {"a3"}; tp has {"a1", "a2"}
            pool_meta=pool_meta,
            query_meta=q_meta,
            query_text="query",
            pool_text_sets=pool_text_sets,
            pool_stats={},
        )

        assert rec["author_jaccard_with_true_positive"] == pytest.approx(0.0)
        assert rec["any_author_overlap_with_true_positive"] is False

    def test_same_domain_flags(self) -> None:
        """same_domain_as_query and same_domain_as_true_positive are correct."""
        from deep_stylometry.experiments.retrieval_inspection import _build_pair_record

        pool_meta = self._make_pool_meta()
        q_meta = self._make_query_meta(tp_pool_idx=0)  # domain="cs"
        pool_text_sets = {i: frozenset() for i in range(3)}

        rec_same = _build_pair_record(
            rank=1, score=0.9, cand_pool_idx=0,  # domain="cs"
            pool_meta=pool_meta, query_meta=q_meta,
            query_text="q", pool_text_sets=pool_text_sets, pool_stats={},
        )
        rec_diff = _build_pair_record(
            rank=2, score=0.7, cand_pool_idx=1,  # domain="phys"
            pool_meta=pool_meta, query_meta=q_meta,
            query_text="q", pool_text_sets=pool_text_sets, pool_stats={},
        )

        assert rec_same["same_domain_as_query"] is True
        assert rec_diff["same_domain_as_query"] is False

    def test_year_diff_computed_correctly(self) -> None:
        """year_diff_vs_true_positive = candidate_year - tp_year."""
        from deep_stylometry.experiments.retrieval_inspection import _build_pair_record

        pool_meta = self._make_pool_meta()
        q_meta = self._make_query_meta(tp_pool_idx=0)  # tp year=2020
        pool_text_sets = {i: frozenset() for i in range(3)}

        rec = _build_pair_record(
            rank=1, score=0.9, cand_pool_idx=1,  # year=2021
            pool_meta=pool_meta, query_meta=q_meta,
            query_text="q", pool_text_sets=pool_text_sets, pool_stats={},
        )

        assert rec["year_diff_vs_true_positive"] == 1  # 2021 - 2020

    def test_unigram_jaccard_with_query(self) -> None:
        """Unigram Jaccard is computed from pre-cached frozensets."""
        from deep_stylometry.experiments.retrieval_inspection import _build_pair_record

        pool_meta = self._make_pool_meta()
        q_meta = self._make_query_meta(tp_pool_idx=0)

        pool_text_sets = {
            0: frozenset(["alpha", "beta", "gamma"]),
            1: frozenset(["delta", "epsilon"]),
            2: frozenset(["alpha", "zeta"]),
        }
        # Query text "alpha beta delta" → frozenset = {"alpha", "beta", "delta"}
        rec = _build_pair_record(
            rank=1, score=0.9, cand_pool_idx=0,  # {"alpha","beta","gamma"}
            pool_meta=pool_meta, query_meta=q_meta,
            query_text="alpha beta delta",
            pool_text_sets=pool_text_sets, pool_stats={},
        )
        # Jaccard({"alpha","beta","gamma"}, {"alpha","beta","delta"}) = 2/4 = 0.5
        assert rec["unigram_jaccard_with_query"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Class 3: Smoke test — full pipeline with mock model/scorer
# ---------------------------------------------------------------------------

class TestSmokePipeline:
    """End-to-end smoke test with mock encoder and MeanInteraction scorer.

    No checkpoint, no HF download, no GPU required.
    """

    def _make_mock_ds(self, n_rows: int = 20, n_unique_halids: int = 5):
        """Build an in-memory HuggingFace dataset."""
        import datasets as hf_datasets

        halids = [f"h{i % n_unique_halids}" for i in range(n_rows)]
        return hf_datasets.Dataset.from_dict(
            {
                "pos_halid": halids,
                "positive": [f"Positive doc content {i}" for i in range(n_rows)],
                "query": [f"Query about topic {i}" for i in range(n_rows)],
                "pos_authorids": [[f"author{i % 3}"] for i in range(n_rows)],
                "query_domain": [f"domain{i % 2}" for i in range(n_rows)],
            }
        )

    def _make_mock_tokenizer(self, hidden: int, seq_len: int = 10):
        """Return a callable mock tokenizer that produces fixed-size tensors."""
        class MockTok:
            pad_token_id = 0
            cls_token_id = 1
            sep_token_id = 2

            def __call__(self, texts, max_length=512, padding=None, truncation=None,
                         add_special_tokens=True, return_tensors=None, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                n = len(texts)
                L = min(max_length, seq_len)
                ids = torch.randint(3, 100, (n, L))
                mask = torch.ones(n, L, dtype=torch.long)
                if return_tensors == "pt":
                    return {"input_ids": ids, "attention_mask": mask}
                return {"input_ids": ids.tolist(), "attention_mask": mask.tolist()}

        return MockTok()

    def _make_mock_model(self, hidden: int):
        """Return a mock encoder that produces (B, S, hidden) embeddings."""
        class MockEncoder(nn.Module):
            def forward(self_, input_ids, attention_mask):
                B, S = input_ids.shape
                return torch.randn(B, S, hidden)
        return MockEncoder()

    def test_pipeline_creates_all_output_files(self, tmp_path: Path) -> None:
        """Full pipeline must create pairs.jsonl, summary.json, report.txt,
        pool_halids.json, config_snapshot.yml in output_dir."""
        from deep_stylometry.experiments.retrieval_inspection import run_pipeline
        from deep_stylometry.modules.mean_interaction import MeanInteraction

        H = 8
        ds = self._make_mock_ds(n_rows=20, n_unique_halids=5)
        model = self._make_mock_model(H)
        scorer = MeanInteraction()
        tokenizer = self._make_mock_tokenizer(H, seq_len=10)

        run_pipeline(
            ds=ds,
            model=model,
            scorer=scorer,
            tokenizer=tokenizer,
            output_dir=tmp_path,
            subset="mock",
            n_seeds=2,
            n_queries_per_seed=3,
            top_k=5,
            batch_size=8,
            chunk_size=4,
            max_length=10,
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_snapshot={"model": {"pooling_method": "mean"}, "data": {}},
            cli_args={"subset": "mock", "n_seeds": 2},
            punct_set=None,
            special_ids=None,
        )

        assert (tmp_path / "pairs.jsonl").exists(), "pairs.jsonl missing"
        assert (tmp_path / "summary.json").exists(), "summary.json missing"
        assert (tmp_path / "report.txt").exists(), "report.txt missing"
        assert (tmp_path / "pool_halids.json").exists(), "pool_halids.json missing"
        assert (tmp_path / "config_snapshot.yml").exists(), "config_snapshot.yml missing"

    def test_pairs_jsonl_has_correct_structure(self, tmp_path: Path) -> None:
        """Each line of pairs.jsonl must have required fields and correct types."""
        from deep_stylometry.experiments.retrieval_inspection import run_pipeline
        from deep_stylometry.modules.mean_interaction import MeanInteraction

        H = 8
        ds = self._make_mock_ds(n_rows=20, n_unique_halids=5)
        model = self._make_mock_model(H)
        scorer = MeanInteraction()
        tokenizer = self._make_mock_tokenizer(H, seq_len=10)

        run_pipeline(
            ds=ds,
            model=model,
            scorer=scorer,
            tokenizer=tokenizer,
            output_dir=tmp_path,
            subset="mock",
            n_seeds=2,
            n_queries_per_seed=3,
            top_k=5,
            batch_size=8,
            chunk_size=4,
            max_length=10,
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_snapshot={},
            cli_args={},
        )

        lines = (tmp_path / "pairs.jsonl").read_text().strip().splitlines()
        assert len(lines) > 0, "pairs.jsonl is empty"

        required_fields = {
            "rank", "score", "pool_idx", "pos_halid",
            "author_jaccard_with_query",
            "author_jaccard_with_true_positive",
            "same_domain_as_query",
            "unigram_jaccard_with_query",
            "unigram_jaccard_with_true_positive",
            "seed", "query_ds_idx", "true_positive_rank",
        }
        for line in lines:
            rec = json.loads(line)
            missing = required_fields - set(rec.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_pool_halids_matches_build_pool(self, tmp_path: Path) -> None:
        """pool_halids.json must list exactly the unique halids, no duplicates."""
        from deep_stylometry.experiments.retrieval_inspection import run_pipeline
        from deep_stylometry.modules.mean_interaction import MeanInteraction

        H = 8
        ds = self._make_mock_ds(n_rows=20, n_unique_halids=5)
        model = self._make_mock_model(H)
        scorer = MeanInteraction()
        tokenizer = self._make_mock_tokenizer(H, seq_len=10)

        run_pipeline(
            ds=ds,
            model=model,
            scorer=scorer,
            tokenizer=tokenizer,
            output_dir=tmp_path,
            subset="mock",
            n_seeds=1,
            n_queries_per_seed=2,
            top_k=3,
            batch_size=8,
            chunk_size=4,
            max_length=10,
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_snapshot={},
            cli_args={},
        )

        pool_halids = json.loads((tmp_path / "pool_halids.json").read_text())
        # 5 unique halids in mock dataset.
        assert len(pool_halids) == 5
        assert len(set(pool_halids)) == len(pool_halids), "Duplicate halids in pool"

    def test_summary_json_has_required_keys(self, tmp_path: Path) -> None:
        """summary.json must have rank_distribution, bucketed_stats, random_baseline."""
        from deep_stylometry.experiments.retrieval_inspection import run_pipeline
        from deep_stylometry.modules.mean_interaction import MeanInteraction

        H = 8
        ds = self._make_mock_ds(n_rows=20, n_unique_halids=5)
        model = self._make_mock_model(H)
        scorer = MeanInteraction()
        tokenizer = self._make_mock_tokenizer(H, seq_len=10)

        run_pipeline(
            ds=ds,
            model=model,
            scorer=scorer,
            tokenizer=tokenizer,
            output_dir=tmp_path,
            subset="mock",
            n_seeds=2,
            n_queries_per_seed=3,
            top_k=5,
            batch_size=8,
            chunk_size=4,
            max_length=10,
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_snapshot={},
            cli_args={},
        )

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert "rank_distribution" in summary
        assert "bucketed_stats" in summary
        assert "random_baseline" in summary
        assert "overall_metrics" in summary
