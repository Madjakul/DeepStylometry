# tests/test_dataset_smoke.py
"""Smoke tests that load a tiny slice of each real dataset.

These tests hit real data sources (HuggingFace Hub or a local ZIP file) and
are therefore marked ``integration``.  They are NOT run by the default test
suite.  Run them explicitly with::

    pytest -m integration tests/test_dataset_smoke.py -v

Requirements
------------
- Network access for HALvest (HuggingFace Hub).
- ``PAN19_ZIP`` environment variable pointing to the local archive, or
  ``PAN19_ROOT`` pointing to an extracted directory, for the PAN 2019 tests.
"""

import os

import pytest


# ---------------------------------------------------------------------------
# HALvest-Contrastive (almanach/halvest-contrastive)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHalvestHubSmoke:
    """Verify that almanach/halvest-contrastive exists and has the right schema."""

    def test_config_names_nonempty(self):
        """The dataset must expose at least one named configuration."""
        from datasets import get_dataset_config_names

        configs = get_dataset_config_names("almanach/halvest-contrastive")
        assert len(configs) > 0, (
            "almanach/halvest-contrastive returned no config names — "
            "the dataset may have been renamed or removed."
        )

    def test_streaming_returns_example(self):
        """Streaming one example from train must not raise and must have correct columns."""
        from datasets import get_dataset_config_names, load_dataset

        configs = get_dataset_config_names("almanach/halvest-contrastive")
        first_config = configs[0]

        ds = load_dataset(
            "almanach/halvest-contrastive",
            name=first_config,
            split="train",
            streaming=True,
        )
        example = next(iter(ds))

        assert "query"    in example, f"Missing 'query' column. Got: {list(example)}"
        assert "positive" in example, f"Missing 'positive' column. Got: {list(example)}"
        assert "negative" in example, f"Missing 'negative' column. Got: {list(example)}"

    def test_query_is_nonempty_string(self):
        """The query field must be a non-empty string."""
        from datasets import get_dataset_config_names, load_dataset

        configs = get_dataset_config_names("almanach/halvest-contrastive")
        ds = load_dataset(
            "almanach/halvest-contrastive",
            name=configs[0],
            split="train",
            streaming=True,
        )
        example = next(iter(ds))

        assert isinstance(example["query"], str) and len(example["query"].strip()) > 0
        assert isinstance(example["positive"], str) and len(example["positive"].strip()) > 0
        assert isinstance(example["negative"], str) and len(example["negative"].strip()) > 0

    def test_dataset_name_default_matches_hub(self):
        """The default halvest_name in dataset_statistics must actually resolve on the Hub."""
        import inspect
        from datasets import get_dataset_config_names
        from deep_stylometry.experiments.dataset_statistics import halvest_statistics

        name = inspect.signature(halvest_statistics).parameters["halvest_name"].default
        # This raises if the dataset doesn't exist on the Hub.
        configs = get_dataset_config_names(name)
        assert len(configs) > 0, f"Default name {name!r} not found on Hub."


# ---------------------------------------------------------------------------
# PAN 2019 — ZIP archive
# ---------------------------------------------------------------------------

_PAN19_ZIP  = os.environ.get("PAN19_ZIP")
_PAN19_ROOT = os.environ.get("PAN19_ROOT")
_pan19_available = bool(_PAN19_ZIP and os.path.isfile(_PAN19_ZIP)) or bool(
    _PAN19_ROOT and os.path.isdir(_PAN19_ROOT)
)


@pytest.mark.integration
@pytest.mark.skipif(
    not _pan19_available,
    reason="PAN 2019 not available — set PAN19_ZIP (path to archive) or PAN19_ROOT (extracted dir)",
)
class TestPan19RealDataSmoke:
    """Load a small slice of the real PAN 2019 dataset."""

    def _parse(self, language: str = "en"):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule

        if _PAN19_ZIP and os.path.isfile(_PAN19_ZIP):
            return PAN19Datamodule._parse_problems_from_zip(_PAN19_ZIP, language=language)
        return PAN19Datamodule._parse_problems(_PAN19_ROOT)

    def test_parses_nonzero_problems(self):
        """English problems must be found — zero results means the ZIP prefix is wrong."""
        problems = self._parse(language="en")
        assert len(problems) > 0, (
            "No English problems parsed from the PAN 2019 archive. "
            "Check that the ZIP has the expected top-level directory prefix."
        )

    def test_unknown_text_nonempty(self):
        """Every parsed problem must have a non-empty unknown text."""
        problems = self._parse(language="en")
        empty = [p["problem_id"] for p in problems if not p["unknown_text"].strip()]
        assert empty == [], f"Problems with empty unknown text: {empty}"

    def test_candidates_nonempty(self):
        """Every parsed problem must have at least one candidate with non-empty text."""
        problems = self._parse(language="en")
        bad = [
            p["problem_id"]
            for p in problems
            if not p["candidates"] or all(not v.strip() for v in p["candidates"].values())
        ]
        assert bad == [], f"Problems with no usable candidate text: {bad}"

    def test_true_author_in_candidates(self):
        """Every problem's true_author must exist in its own candidates dict."""
        problems = self._parse(language="en")
        bad = [
            (p["problem_id"], p.get("unknown_file", "?"), p["true_author"])
            for p in problems
            if p["true_author"] not in p["candidates"]
        ]
        assert bad == [], f"true_author missing from candidates: {bad[:5]}"

    def test_required_keys_present(self):
        """Each problem dict must have the keys the rest of the pipeline expects."""
        problems = self._parse(language="en")
        required = {"problem_id", "unknown_text", "candidates", "true_author"}
        for p in problems[:10]:  # spot-check first 10
            missing = required - set(p.keys())
            assert not missing, f"{p['problem_id']} missing keys: {missing}"

    def test_zip_based_parser_has_unknown_file_key(self):
        """The ZIP parser must emit 'unknown_file' (used by _apply_split)."""
        if not (_PAN19_ZIP and os.path.isfile(_PAN19_ZIP)):
            pytest.skip("ZIP-based parser only; PAN19_ZIP not available")

        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule

        problems = PAN19Datamodule._parse_problems_from_zip(_PAN19_ZIP, language="en")
        assert all("unknown_file" in p for p in problems[:10])

    def test_pan19_statistics_with_real_data(self):
        """pan19_statistics() must return non-trivial counts on real data."""
        from deep_stylometry.experiments.dataset_statistics import pan19_statistics

        kwargs = (
            {"pan19_zip": _PAN19_ZIP} if _PAN19_ZIP and os.path.isfile(_PAN19_ZIP)
            else {"pan19_root": _PAN19_ROOT}
        )
        stats = pan19_statistics(**kwargs, language="en")

        assert stats["n_problems"] > 0,      f"n_problems=0: {stats}"
        assert stats["n_queries_total"] > 0,  f"n_queries_total=0: {stats}"
        assert stats["unknown_text_word_length"]["mean"] > 0
        assert stats["candidate_text_word_length"]["mean"] > 0
