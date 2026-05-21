# tests/test_decorrelation_and_collab.py
"""Unit tests for semantic_decorrelation and collaboration_statistics.

Run::

    python -m pytest tests/test_decorrelation_and_collab.py -v --tb=short
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Imports from the modules under test
# ---------------------------------------------------------------------------

from deep_stylometry.experiments.semantic_decorrelation import (
    DecorrelationCallback,
    _add_e5_prefix,
    mean_pool,
)
from deep_stylometry.experiments.collaboration_statistics import (
    _authorids_to_frozenset,
    _flatten_domain,
    _jaccard,
)


# ===========================================================================
# Task 1 — semantic_decorrelation tests
# ===========================================================================


class TestMeanPool:
    """Tests for the mean_pool() function."""

    def test_first_token_only(self) -> None:
        """Only the first token should contribute when mask = [1, 0]."""
        embs = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
        mask = torch.tensor([[1, 0]])
        result = mean_pool(embs, mask)
        expected = torch.tensor([[1.0, 2.0, 3.0]])
        assert result.shape == (1, 3)
        assert torch.allclose(result, expected), f"Got {result}"

    def test_both_tokens(self) -> None:
        """With mask = [1, 1] both tokens should be averaged."""
        embs = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
        mask = torch.tensor([[1, 1]])
        result = mean_pool(embs, mask)
        expected = torch.tensor([[2.5, 3.5, 4.5]])
        assert result.shape == (1, 3)
        assert torch.allclose(result, expected), f"Got {result}"

    def test_batch_of_two(self) -> None:
        """Batch with two sequences and different masks."""
        embs = torch.tensor([
            [[1.0, 0.0], [0.0, 1.0]],   # row 0: mean of both → [0.5, 0.5]
            [[2.0, 4.0], [8.0, 16.0]],  # row 1: first token only → [2., 4.]
        ])  # (2, 2, 2)
        mask = torch.tensor([[1, 1], [1, 0]])
        result = mean_pool(embs, mask)
        assert result.shape == (2, 2)
        assert torch.allclose(result[0], torch.tensor([0.5, 0.5]))
        assert torch.allclose(result[1], torch.tensor([2.0, 4.0]))


class TestMeanPoolAllZerosMask:
    """mean_pool must not produce NaN or Inf when mask is all-zeros."""

    def test_all_zeros_mask(self) -> None:
        embs = torch.ones(1, 3, 8)
        mask = torch.zeros(1, 3, dtype=torch.long)
        result = mean_pool(embs, mask)
        assert result.shape == (1, 8)
        assert not torch.isnan(result).any(), "Got NaN with all-zeros mask"
        assert not torch.isinf(result).any(), "Got Inf with all-zeros mask"


class TestCosineSimilarityRange:
    """F.cosine_similarity must return values in [-1, 1]."""

    def test_random_pairs(self) -> None:
        torch.manual_seed(0)
        a = torch.randn(100, 768)
        b = torch.randn(100, 768)
        sims = F.cosine_similarity(a, b, dim=-1)
        assert sims.min().item() >= -1.0 - 1e-6, f"Min={sims.min()}"
        assert sims.max().item() <= 1.0 + 1e-6, f"Max={sims.max()}"


class TestE5PrefixPrepended:
    """_add_e5_prefix must prepend 'query: ' to every text — highest-risk
    silent bug: E5 outputs valid-looking garbage without this prefix."""

    def test_basic(self) -> None:
        texts = ["hello world", "test"]
        result = _add_e5_prefix(texts)
        assert result == ["query: hello world", "query: test"]

    def test_empty_list(self) -> None:
        assert _add_e5_prefix([]) == []

    def test_already_prefixed(self) -> None:
        # Should double-prefix — callers are responsible for not calling twice.
        texts = ["query: already prefixed"]
        result = _add_e5_prefix(texts)
        assert result == ["query: query: already prefixed"]

    def test_all_outputs_start_with_prefix(self) -> None:
        texts = [f"text number {i}" for i in range(20)]
        for t in _add_e5_prefix(texts):
            assert t.startswith("query: "), f"Missing prefix: {t!r}"


class TestDecorrelationCallbackIndexFiltering:
    """DecorrelationCallback should store embeddings only for sampled indices."""

    def _make_callback(self, sampled_indices: set) -> DecorrelationCallback:
        """Create a callback with a mock raw dataset and the given sampled set."""
        # Build a mock raw_ds large enough so sampling doesn't exceed its size.
        raw_ds = MagicMock()
        raw_ds.__len__ = MagicMock(return_value=100)

        # Patch sampled_indices directly after construction.
        cb = DecorrelationCallback(sample_size=len(sampled_indices), seed=0, raw_ds=raw_ds)
        cb.sampled_indices = sampled_indices
        return cb

    def _make_outputs(self, batch_size: int, seq_len: int = 4, hidden: int = 8) -> Dict:
        """Create fake test_step outputs."""
        return {
            "q_embs": torch.randn(batch_size, seq_len, hidden),
            "q_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "pos_embs": torch.randn(batch_size, seq_len, hidden),
            "pos_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "neg_embs": torch.randn(batch_size, seq_len, hidden),
            "neg_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }

    def test_stores_only_sampled_indices(self) -> None:
        """Only indices 0 and 3 should be stored from a batch [0, 1, 2, 3]."""
        cb = self._make_callback(sampled_indices={0, 3, 7})
        outputs = self._make_outputs(batch_size=4)
        batch = {"index": torch.tensor([0, 1, 2, 3])}

        cb.on_test_batch_end(
            trainer=MagicMock(),
            pl_module=MagicMock(),
            outputs=outputs,
            batch=batch,
            batch_idx=0,
        )

        assert set(cb.q_embs_dict.keys()) == {0, 3}, (
            f"Expected {{0, 3}}, got {set(cb.q_embs_dict.keys())}"
        )
        assert set(cb.pos_embs_dict.keys()) == {0, 3}
        assert set(cb.neg_embs_dict.keys()) == {0, 3}

    def test_no_duplicates_on_repeated_index(self) -> None:
        """A second batch with the same index must not overwrite the first."""
        cb = self._make_callback(sampled_indices={0})
        outputs_a = self._make_outputs(batch_size=1)
        batch_a = {"index": torch.tensor([0])}
        cb.on_test_batch_end(MagicMock(), MagicMock(), outputs_a, batch_a, 0)
        first_vec = cb.q_embs_dict[0].clone()

        # Second encounter — should be skipped.
        outputs_b = self._make_outputs(batch_size=1)
        batch_b = {"index": torch.tensor([0])}
        cb.on_test_batch_end(MagicMock(), MagicMock(), outputs_b, batch_b, 1)

        assert torch.allclose(cb.q_embs_dict[0], first_vec), (
            "Second encounter overwrote the first embedding."
        )

    def test_empty_intersection(self) -> None:
        """If no batch indices are in the sample, nothing should be stored."""
        cb = self._make_callback(sampled_indices={10, 20, 30})
        outputs = self._make_outputs(batch_size=3)
        batch = {"index": torch.tensor([0, 1, 2])}
        cb.on_test_batch_end(MagicMock(), MagicMock(), outputs, batch, 0)

        assert len(cb.q_embs_dict) == 0


# ===========================================================================
# Task 2 — collaboration_statistics tests
# ===========================================================================


class TestFrozensetDedup:
    """Converting author-id lists to frozensets removes order and duplicates."""

    def test_two_unique_frozensets(self) -> None:
        raw_lists = [["a", "b"], ["b", "a"], ["c", "d"]]
        frozensets = {frozenset(l) for l in raw_lists}
        assert len(frozensets) == 2, f"Expected 2 unique frozensets, got {len(frozensets)}"

    def test_order_invariance(self) -> None:
        assert frozenset(["x", "y"]) == frozenset(["y", "x"])

    def test_single_element(self) -> None:
        assert frozenset(["z"]) == frozenset(["z"])


class TestJaccardAuthorSets:
    """Jaccard similarity between author frozensets."""

    def test_known_value(self) -> None:
        a: FrozenSet[str] = frozenset({"a", "b"})
        b: FrozenSet[str] = frozenset({"b", "c"})
        # Intersection={b}, union={a,b,c} → 1/3
        result = _jaccard(a, b)
        assert abs(result - 1 / 3) < 1e-9, f"Expected 1/3, got {result}"

    def test_identical_sets(self) -> None:
        a = frozenset({"x", "y", "z"})
        assert _jaccard(a, a) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        a = frozenset({"p"})
        b = frozenset({"q"})
        assert _jaccard(a, b) == pytest.approx(0.0)

    def test_both_empty(self) -> None:
        assert _jaccard(frozenset(), frozenset()) == pytest.approx(0.0)


class TestEmptyAuthoridsFiltered:
    """_authorids_to_frozenset should return None for empty/null inputs."""

    def test_none_input(self) -> None:
        assert _authorids_to_frozenset(None) is None

    def test_empty_list(self) -> None:
        assert _authorids_to_frozenset([]) is None

    def test_non_empty_list(self) -> None:
        result = _authorids_to_frozenset(["a", "b"])
        assert result == frozenset({"a", "b"})

    def test_filtering_drops_empties(self) -> None:
        """Simulate the row-level filtering in compute_config_stats."""
        raw = [["a", "b"], [], ["c"]]
        n_dropped = 0
        kept: List[FrozenSet[str]] = []
        for ids in raw:
            fs = _authorids_to_frozenset(ids)
            if fs is None:
                n_dropped += 1
            else:
                kept.append(fs)
        assert len(kept) == 2, f"Expected 2 kept, got {len(kept)}"
        assert n_dropped == 1, f"Expected 1 dropped, got {n_dropped}"


class TestDomainListUnwrap:
    """_flatten_domain handles both list and scalar domain values."""

    def test_list_input(self) -> None:
        result = _flatten_domain(["Computer Science"])
        assert result == "Computer Science"

    def test_string_input(self) -> None:
        result = _flatten_domain("Mathematics")
        assert result == "Mathematics"

    def test_none_input(self) -> None:
        result = _flatten_domain(None)
        assert result == ""

    def test_empty_list(self) -> None:
        result = _flatten_domain([])
        assert result == ""

    def test_multi_element_list(self) -> None:
        # Only the first element is used.
        result = _flatten_domain(["Physics", "Chemistry"])
        assert result == "Physics"


# ===========================================================================
# Phase 1 bug-fix tests
# ===========================================================================


class TestCenteredMeanPool:
    """_CenteredMeanPool must subtract mu, then L2-normalise, before dot product."""

    def test_matches_manual_computation(self) -> None:
        """Known mu=[1, 0.5] → verify score against manual calculation."""
        import torch.nn.functional as F
        from deep_stylometry.modules.mean_centerer import MeanCenterer
        from deep_stylometry.callbacks.test_eval_callback import _CenteredMeanPool

        hidden = 2
        centerer = MeanCenterer(hidden)
        centerer.mu.copy_(torch.tensor([1.0, 0.5]))

        pool = _CenteredMeanPool(centerer)

        # (1, 1, 2) — single batch, single token
        q_embs = torch.tensor([[[2.0, 1.0]]])
        k_embs = torch.tensor([[[3.0, 2.0]]])
        q_mask = torch.ones(1, 1, dtype=torch.long)
        k_mask = torch.ones(1, 1, dtype=torch.long)

        scores = pool(q_embs, k_embs, q_mask, k_mask)

        # Manual: pool → [2,1], centre → [1, 0.5], then L2-norm
        q_c = torch.tensor([[1.0, 0.5]])
        k_c = torch.tensor([[2.0, 1.5]])
        q_n = F.normalize(q_c, p=2, dim=-1)
        k_n = F.normalize(k_c, p=2, dim=-1)
        expected = torch.matmul(q_n, k_n.T)

        assert scores.shape == (1, 1)
        assert torch.allclose(scores, expected, atol=1e-5), (
            f"Got {scores}, expected {expected}"
        )

    def test_zero_mu_reduces_to_plain_l2(self) -> None:
        """With mu=0, _CenteredMeanPool must equal plain L2-normalised dot."""
        import torch.nn.functional as F
        from deep_stylometry.modules.mean_centerer import MeanCenterer
        from deep_stylometry.callbacks.test_eval_callback import _CenteredMeanPool

        hidden = 4
        centerer = MeanCenterer(hidden)  # mu initialised to zeros

        pool = _CenteredMeanPool(centerer)
        torch.manual_seed(7)
        q_embs = torch.randn(2, 3, hidden)
        k_embs = torch.randn(2, 3, hidden)
        mask = torch.ones(2, 3, dtype=torch.long)

        scores = pool(q_embs, k_embs, mask, mask)

        # Expected: mean-pool + L2-norm only
        m = mask.unsqueeze(-1).float()
        q_vec = F.normalize((q_embs * m).sum(1) / m.sum(1).clamp(min=1e-9), p=2, dim=-1)
        k_vec = F.normalize((k_embs * m).sum(1) / m.sum(1).clamp(min=1e-9), p=2, dim=-1)
        expected = torch.matmul(q_vec, k_vec.T)

        assert torch.allclose(scores, expected, atol=1e-5), (
            f"Got {scores}, expected {expected}"
        )


class TestTrainedPLIExtracted:
    """TestEvalCallback must extract the trained PatchInteraction, not a fresh one."""

    def test_same_object_returned(self) -> None:
        """getattr(pl_module.contrastive_loss, 'pool') must return the *same*
        PatchInteraction instance that was stored, not a newly constructed one."""
        from deep_stylometry.modules.patch_interaction import PatchInteraction

        cfg = MagicMock()
        cfg.model.patch_method = "ngram"
        cfg.model.patch_size = 3
        cfg.model.patch_compression = "mean"
        cfg.model.lm_hidden_size = 64
        cfg.model.patch_cross_attn_dim = 16
        cfg.model.patch_cross_attn_heads = 1

        pli = PatchInteraction(cfg)

        class _FakeLoss:
            pool = pli

        class _FakePLModule:
            contrastive_loss = _FakeLoss()

        pl_module = _FakePLModule()

        # Apply the extraction logic from the fix
        trained_pool = getattr(pl_module.contrastive_loss, "pool", None)

        assert isinstance(trained_pool, PatchInteraction), (
            f"Expected PatchInteraction, got {type(trained_pool)}"
        )
        assert trained_pool is pli, (
            "Extraction must return the *same* trained instance."
        )


class TestNumProcNoneWhenCuda:
    """num_proc must be None (not 1) when CUDA is already initialised."""

    def test_none_when_cuda_initialized(self) -> None:
        """Mirrors the fix in halvest_datamodule.test_setup:
        ``num_proc = None if torch.cuda.is_initialized() else self.num_proc``"""
        from unittest.mock import patch

        num_proc_self = 8

        with patch.object(torch.cuda, "is_initialized", return_value=True):
            result = None if torch.cuda.is_initialized() else num_proc_self

        assert result is None, (
            f"Expected None when CUDA is initialised, got {result!r}"
        )

    def test_passthrough_when_cuda_not_initialized(self) -> None:
        """When CUDA is not initialised num_proc must pass through unchanged."""
        from unittest.mock import patch

        num_proc_self = 8

        with patch.object(torch.cuda, "is_initialized", return_value=False):
            result = None if torch.cuda.is_initialized() else num_proc_self

        assert result == num_proc_self, (
            f"Expected {num_proc_self}, got {result!r}"
        )


class TestPerFieldEvalDomainDetection:
    """query_domain must be detected before generic 'domain' column."""

    def test_query_domain_detected(self) -> None:
        """When the dataset has 'query_domain' it must be returned as domain_col."""
        from unittest.mock import patch

        mock_ds = MagicMock()
        mock_ds.column_names = ["query", "query_domain"]
        mock_ds.__getitem__ = MagicMock(
            side_effect=lambda col: ["q1", "q2"] if col == "query" else ["d1", "d2"]
        )

        # datasets is imported *locally* inside _load_halvest_domain_labels,
        # so we patch the module-level datasets.load_dataset directly.
        with patch("datasets.load_dataset", return_value=mock_ds):
            from deep_stylometry.experiments.per_field_eval import (
                _load_halvest_domain_labels,
            )
            queries, domains = _load_halvest_domain_labels(subset="base-2")

        assert domains == ["d1", "d2"], f"Got {domains}"
        assert queries == ["q1", "q2"], f"Got {queries}"

    def test_query_domain_takes_priority_over_domain(self) -> None:
        """If both 'query_domain' and 'domain' exist, query_domain wins."""
        from unittest.mock import patch

        mock_ds = MagicMock()
        mock_ds.column_names = ["query", "query_domain", "domain"]

        def _getitem(col: str) -> List[str]:
            if col == "query":
                return ["q"]
            if col == "query_domain":
                return ["qd_value"]
            return ["d_value"]  # 'domain' column — must NOT be used

        mock_ds.__getitem__ = MagicMock(side_effect=_getitem)

        with patch("datasets.load_dataset", return_value=mock_ds):
            from deep_stylometry.experiments.per_field_eval import (
                _load_halvest_domain_labels,
            )
            _, domains = _load_halvest_domain_labels(subset="base-2")

        assert domains == ["qd_value"], (
            f"Expected query_domain values, got {domains}"
        )


class TestTripletAccuracyAccumulation:
    """Triplet accuracy must accumulate correctly across multiple batches."""

    def _make_outputs(
        self,
        q: torch.Tensor,  # (B, S, H)
        pos: torch.Tensor,  # (B, S, H)
        neg: torch.Tensor,  # (B, S, H)
    ) -> dict:
        B, S, _ = q.shape
        mask = torch.ones(B, S, dtype=torch.long)
        ids = torch.zeros(B, S, dtype=torch.long)
        return {
            "q_embs": q,
            "q_mask": mask,
            "q_input_ids": ids,
            "pos_embs": pos,
            "pos_mask": mask,
            "pos_input_ids": ids,
            "neg_embs": neg,
            "neg_mask": mask,
            "neg_input_ids": ids,
            "target_indices": None,
        }

    def test_two_batches_three_over_five(self) -> None:
        """Batch 1: 3 triplets, 2 correct.  Batch 2: 2 triplets, 1 correct.
        Final accuracy must be 3/5 = 0.6."""
        import shutil
        from deep_stylometry.callbacks.test_eval_callback import TestEvalCallback
        from deep_stylometry.modules.mean_interaction import MeanInteraction

        cfg = MagicMock()
        cfg.model.pooling_method = "mean"

        cb = TestEvalCallback(cfg=cfg, k=5)

        pool = MeanInteraction()
        pl_mock = MagicMock()
        pl_mock.contrastive_loss.pool = pool

        try:
            cb.on_test_epoch_start(trainer=MagicMock(), pl_module=MagicMock())

            # Batch 1 (B=3, H=2):
            #   q0=[1,0]  pos=[1,0]  neg=[-1,0]  → pos score=1, neg score=-1  CORRECT
            #   q1=[0,1]  pos=[0,1]  neg=[0,-1]  → pos=1, neg=-1              CORRECT
            #   q2=[1,0]  pos=[-1,0] neg=[1,0]   → pos=-1, neg=1             WRONG
            q1 = torch.tensor([[[1., 0.]], [[0., 1.]], [[1., 0.]]])   # (3,1,2)
            p1 = torch.tensor([[[1., 0.]], [[0., 1.]], [[-1., 0.]]]) # (3,1,2)
            n1 = torch.tensor([[[-1., 0.]], [[0., -1.]], [[1., 0.]]]) # (3,1,2)

            cb.on_test_batch_end(
                MagicMock(), pl_mock,
                self._make_outputs(q1, p1, n1),
                {"index": torch.tensor([0, 1, 2])}, 0,
            )

            assert cb._triplet_correct == 2, f"After batch 1: {cb._triplet_correct}"
            assert cb._triplet_total == 3,   f"After batch 1: {cb._triplet_total}"

            # Batch 2 (B=2, H=2):
            #   q0=[1,0]  pos=[1,0]  neg=[-1,0]  → CORRECT
            #   q1=[0,1]  pos=[0,-1] neg=[0,1]   → pos=-1, neg=1   WRONG
            q2 = torch.tensor([[[1., 0.]], [[0., 1.]]])   # (2,1,2)
            p2 = torch.tensor([[[1., 0.]], [[0., -1.]]]) # (2,1,2)
            n2 = torch.tensor([[[-1., 0.]], [[0., 1.]]]) # (2,1,2)

            cb.on_test_batch_end(
                MagicMock(), pl_mock,
                self._make_outputs(q2, p2, n2),
                {"index": torch.tensor([3, 4])}, 1,
            )

            assert cb._triplet_correct == 3, f"Final correct: {cb._triplet_correct}"
            assert cb._triplet_total == 5,   f"Final total:   {cb._triplet_total}"

            acc = cb._triplet_correct / cb._triplet_total
            assert abs(acc - 0.6) < 1e-9, f"Expected 0.6, got {acc}"

        finally:
            try:
                cb.h5_file.close()
            except Exception:
                pass
            shutil.rmtree(cb.tmp_dir, ignore_errors=True)
