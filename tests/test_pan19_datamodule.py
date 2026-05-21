# tests/test_pan19_datamodule.py
"""Tests for PAN19Datamodule with synthetic data.

All tests are self-contained and use either synthetic directory trees or
synthetic ZIP archives — no network access, no GPU required.
"""

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, List

import pytest

from deep_stylometry.utils.configs import BaseConfig


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def pan19_cfg():
    """Minimal BaseConfig for PAN19Datamodule tests."""
    cfg = BaseConfig()
    cfg.data.ds_name = "pan19"
    cfg.data.batch_size = 4
    cfg.data.tokenizer_name = "answerdotai/ModernBERT-base"
    cfg.data.max_length = 64
    cfg.data.padding = "do_not_pad"
    cfg.data.truncation = "longest_first"
    cfg.data.add_special_tokens = True
    cfg.data.load_from_cache_file = False
    cfg.train.precision = "32"
    return cfg


# ---------------------------------------------------------------------------
# Helpers: synthetic directory format (for _parse_problems)
# ---------------------------------------------------------------------------


def _write_dir_problem(
    root: Path,
    problem_id: str,
    candidates: Dict[str, List[str]],
    unknown_text: str,
    true_author: str,
    encoding: str = "utf-8",
) -> None:
    """Write a synthetic PAN 2019 problem directory (legacy format)."""
    prob_dir = root / problem_id
    prob_dir.mkdir(parents=True)

    (prob_dir / "unknown.txt").write_text(unknown_text, encoding=encoding)

    for cand_id, known_texts in candidates.items():
        cand_dir = prob_dir / cand_id
        cand_dir.mkdir()
        for i, text in enumerate(known_texts, start=1):
            (cand_dir / f"known{i:02d}.txt").write_text(text, encoding=encoding)

    # Legacy single-unknown format: key is "true-author" (HYPHEN)
    (prob_dir / "ground-truth.json").write_text(
        json.dumps({"true-author": true_author}), encoding="utf-8"
    )


@pytest.fixture
def synthetic_pan19_root(tmp_path):
    """Create a synthetic PAN 2019 directory with 3 problems (legacy format)."""
    root = tmp_path / "pan19"
    root.mkdir()
    _write_dir_problem(
        root, "problem00001",
        candidates={
            "candidate001": ["known text A1.", "known text A2."],
            "candidate002": ["known text B1."],
            "candidate003": ["known text C1.", "known text C2.", "known text C3."],
        },
        unknown_text="This is the unknown document for problem 1.",
        true_author="candidate002",
    )
    _write_dir_problem(
        root, "problem00002",
        candidates={
            "candidate001": ["known text D1."],
            "candidate002": ["known text E1.", "known text E2."],
        },
        unknown_text="Another unknown document for problem 2.",
        true_author="candidate001",
    )
    _write_dir_problem(
        root, "problem00003",
        candidates={
            "candidate001": ["known text F1."],
            "candidate002": ["known text G1."],
            "candidate003": ["known text H1."],
            "candidate004": ["known text I1."],
        },
        unknown_text="Yet another unknown document for problem 3.",
        true_author="candidate003",
    )
    return str(root)


# ---------------------------------------------------------------------------
# Helpers: synthetic ZIP format (for _parse_problems_from_zip)
# ---------------------------------------------------------------------------

_ZIP_PREFIX = "pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23/"


def _build_synthetic_zip(tmp_path: Path, problems: List[Dict]) -> str:
    """Build a minimal PAN 2019 ZIP archive in *tmp_path*.

    Each entry in *problems*:
      - problem_id (str)
      - language (str) — e.g. "en"
      - candidates: {cand_id: [known_text, ...]}
      - unknowns: {filename: {"text": str, "true_author": str|"<UNK>"}}
    """
    zip_path = str(tmp_path / "pan19.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        # collection-info.json
        collection = [
            {"problem-name": p["problem_id"], "language": p["language"], "encoding": "UTF-8"}
            for p in problems
        ]
        zf.writestr(f"{_ZIP_PREFIX}collection-info.json", json.dumps(collection))

        for prob in problems:
            pid = prob["problem_id"]
            pp = f"{_ZIP_PREFIX}{pid}/"

            # Candidates
            for cand_id, known_texts in prob["candidates"].items():
                for i, text in enumerate(known_texts, start=1):
                    zf.writestr(f"{pp}{cand_id}/known{i:02d}.txt", text)

            # Unknowns + ground truth
            # Mirror the real archive layout: unknowns live in an unknown/ subdir.
            gt_list = []
            for unk_file, info in prob["unknowns"].items():
                zf.writestr(f"{pp}unknown/{unk_file}", info["text"])
                gt_list.append({"unknown-text": unk_file, "true-author": info["true_author"]})
            zf.writestr(f"{pp}ground-truth.json", json.dumps({"ground_truth": gt_list}))

    return zip_path


@pytest.fixture
def synthetic_pan19_zip(tmp_path):
    """Synthetic PAN 2019 ZIP with 2 English problems, multiple unknowns each."""
    problems = [
        {
            "problem_id": "problem00001",
            "language": "en",
            "candidates": {
                "candidate00001": ["Known A1.", "Known A2."],
                "candidate00002": ["Known B1."],
                "candidate00003": ["Known C1.", "Known C2."],
            },
            "unknowns": {
                "unknown00001.txt": {"text": "Unknown query 1.", "true_author": "candidate00001"},
                "unknown00002.txt": {"text": "Unknown query 2.", "true_author": "candidate00002"},
                "unknown00003.txt": {"text": "Unknown query 3.", "true_author": "<UNK>"},
                "unknown00004.txt": {"text": "Unknown query 4.", "true_author": "candidate00003"},
                "unknown00005.txt": {"text": "Unknown query 5.", "true_author": "candidate00001"},
            },
        },
        {
            "problem_id": "problem00002",
            "language": "en",
            "candidates": {
                "candidate00001": ["Known D1."],
                "candidate00002": ["Known E1.", "Known E2."],
            },
            "unknowns": {
                "unknown00001.txt": {"text": "Another unknown 1.", "true_author": "candidate00001"},
                "unknown00002.txt": {"text": "Another unknown 2.", "true_author": "candidate00002"},
                "unknown00003.txt": {"text": "Another unknown 3.", "true_author": "<UNK>"},
            },
        },
        {
            "problem_id": "problem00003",
            "language": "fr",  # French — should be excluded with language="en"
            "candidates": {"candidate00001": ["Texte français."]},
            "unknowns": {
                "unknown00001.txt": {"text": "Inconnu.", "true_author": "candidate00001"},
            },
        },
    ]
    return _build_synthetic_zip(tmp_path, problems)


# ---------------------------------------------------------------------------
# Tests: _parse_problems (legacy directory format)
# ---------------------------------------------------------------------------


class TestParseProblemsDirectory:
    def test_all_problems_found(self, synthetic_pan19_root):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems(synthetic_pan19_root)
        assert len(problems) == 3

    def test_problem_ids(self, synthetic_pan19_root):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems(synthetic_pan19_root)
        ids = {p["problem_id"] for p in problems}
        assert ids == {"problem00001", "problem00002", "problem00003"}

    def test_unknown_text_loaded(self, synthetic_pan19_root):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems(synthetic_pan19_root)
        prob1 = next(p for p in problems if p["problem_id"] == "problem00001")
        assert "problem 1" in prob1["unknown_text"]

    def test_candidates_concatenated(self, synthetic_pan19_root):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems(synthetic_pan19_root)
        prob1 = next(p for p in problems if p["problem_id"] == "problem00001")
        assert "candidate001" in prob1["candidates"]
        assert "known text A1." in prob1["candidates"]["candidate001"]
        assert "known text A2." in prob1["candidates"]["candidate001"]

    def test_true_author_loaded(self, synthetic_pan19_root):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems(synthetic_pan19_root)
        prob1 = next(p for p in problems if p["problem_id"] == "problem00001")
        assert prob1["true_author"] == "candidate002"

    def test_missing_ground_truth_logged(self, tmp_path, caplog):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        root = tmp_path / "pan19_ngt"
        root.mkdir()
        prob_dir = root / "problem00001"
        prob_dir.mkdir()
        (prob_dir / "unknown.txt").write_text("unknown text")
        cand_dir = prob_dir / "candidate001"
        cand_dir.mkdir()
        (cand_dir / "known01.txt").write_text("known text")
        # No ground-truth.json
        with caplog.at_level(logging.WARNING):
            problems = PAN19Datamodule._parse_problems(str(root))
        assert len(problems) == 1
        assert problems[0]["true_author"] is None
        assert any("ground-truth" in m.lower() for m in caplog.messages)

    def test_empty_unknown_skipped(self, tmp_path, caplog):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        root = tmp_path / "pan19_empty"
        root.mkdir()
        prob_dir = root / "problem00001"
        prob_dir.mkdir()
        (prob_dir / "unknown.txt").write_text("   \n   ")
        cand_dir = prob_dir / "candidate001"
        cand_dir.mkdir()
        (cand_dir / "known01.txt").write_text("known text")
        (prob_dir / "ground-truth.json").write_text(
            json.dumps({"true-author": "candidate001"})
        )
        with caplog.at_level(logging.WARNING):
            problems = PAN19Datamodule._parse_problems(str(root))
        assert len(problems) == 0

    def test_encoding_fallback(self, tmp_path):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        root = tmp_path / "pan19_latin"
        root.mkdir()
        _write_dir_problem(
            root, "problem00001",
            candidates={"candidate001": ["Über schöne Dinge."]},
            unknown_text="Der unbekannte Text: Ärger.",
            true_author="candidate001",
            encoding="latin-1",
        )
        problems = PAN19Datamodule._parse_problems(str(root))
        assert len(problems) == 1
        assert problems[0]["unknown_text"]  # non-empty


# ---------------------------------------------------------------------------
# Tests: _parse_problems_from_zip (actual data format)
# ---------------------------------------------------------------------------


class TestParseProblemsFromZip:
    def test_filters_to_english_only(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems_from_zip(
            synthetic_pan19_zip, language="en"
        )
        pids = {p["problem_id"] for p in problems}
        assert "problem00003" not in pids, "French problem should be excluded"
        assert "problem00001" in pids
        assert "problem00002" in pids

    def test_skips_unk_true_authors(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems_from_zip(
            synthetic_pan19_zip, language="en"
        )
        for p in problems:
            assert p["true_author"] != "<UNK>", "UNK entries should be excluded"

    def test_multiple_unknowns_per_problem(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems_from_zip(
            synthetic_pan19_zip, language="en"
        )
        # problem00001 has 5 unknowns; 1 is <UNK> → 4 valid
        p1 = [p for p in problems if p["problem_id"] == "problem00001"]
        assert len(p1) == 4

    def test_known_texts_concatenated(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems_from_zip(
            synthetic_pan19_zip, language="en"
        )
        p1 = next(p for p in problems if p["problem_id"] == "problem00001")
        # candidate00001 has 2 known files
        assert "Known A1." in p1["candidates"]["candidate00001"]
        assert "Known A2." in p1["candidates"]["candidate00001"]

    def test_true_author_correct(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems_from_zip(
            synthetic_pan19_zip, language="en"
        )
        # unknown00001.txt for problem00001 → candidate00001
        p1_unk1 = next(
            p for p in problems
            if p["problem_id"] == "problem00001" and p["unknown_file"] == "unknown00001.txt"
        )
        assert p1_unk1["true_author"] == "candidate00001"

    def test_unknown_file_field_present(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems_from_zip(
            synthetic_pan19_zip, language="en"
        )
        for p in problems:
            assert "unknown_file" in p


# ---------------------------------------------------------------------------
# Tests: dev/test split
# ---------------------------------------------------------------------------


class TestDevTestSplit:
    def test_split_sizes(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        all_probs = PAN19Datamodule._parse_problems_from_zip(synthetic_pan19_zip, language="en")
        # problem00001: 4 valid, problem00002: 2 valid → total 6
        dev = PAN19Datamodule._apply_split(all_probs, "dev", test_fraction=0.2)
        test = PAN19Datamodule._apply_split(all_probs, "test", test_fraction=0.2)
        assert len(dev) + len(test) == len(all_probs)
        assert len(test) >= 1

    def test_dev_test_disjoint(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        all_probs = PAN19Datamodule._parse_problems_from_zip(synthetic_pan19_zip, language="en")
        dev = PAN19Datamodule._apply_split(all_probs, "dev", test_fraction=0.2)
        test = PAN19Datamodule._apply_split(all_probs, "test", test_fraction=0.2)
        dev_keys = {(p["problem_id"], p.get("unknown_file")) for p in dev}
        test_keys = {(p["problem_id"], p.get("unknown_file")) for p in test}
        assert dev_keys.isdisjoint(test_keys), "Dev and test sets overlap"

    def test_none_split_returns_all(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        all_probs = PAN19Datamodule._parse_problems_from_zip(synthetic_pan19_zip, language="en")
        result = PAN19Datamodule._apply_split(all_probs, None, test_fraction=0.2)
        assert len(result) == len(all_probs)

    def test_split_is_deterministic(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        all_probs = PAN19Datamodule._parse_problems_from_zip(synthetic_pan19_zip, language="en")
        test1 = PAN19Datamodule._apply_split(all_probs, "test", test_fraction=0.2)
        test2 = PAN19Datamodule._apply_split(all_probs, "test", test_fraction=0.2)
        assert [p["unknown_file"] for p in test1] == [p["unknown_file"] for p in test2]


# ---------------------------------------------------------------------------
# Tests: _convert_to_triplets
# ---------------------------------------------------------------------------


class TestConvertToTriplets:
    def test_triplet_counts_match_valid_queries(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems_from_zip(synthetic_pan19_zip, language="en")
        ds = PAN19Datamodule._convert_to_triplets(problems, split=None)
        assert len(ds) == len(problems)

    def test_positive_is_true_author(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems_from_zip(synthetic_pan19_zip, language="en")
        ds = PAN19Datamodule._convert_to_triplets(problems, split=None)
        for i, row in enumerate(ds):
            prob = problems[i]
            assert row["positive"] == prob["candidates"][prob["true_author"]]

    def test_negative_differs_from_positive(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems_from_zip(synthetic_pan19_zip, language="en")
        ds = PAN19Datamodule._convert_to_triplets(problems, split=None)
        for row in ds:
            assert row["positive"] != row["negative"]

    def test_deterministic_negative(self, synthetic_pan19_zip):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = PAN19Datamodule._parse_problems_from_zip(synthetic_pan19_zip, language="en")
        ds1 = PAN19Datamodule._convert_to_triplets(problems, seed=42, split=None)
        ds2 = PAN19Datamodule._convert_to_triplets(problems, seed=42, split=None)
        for r1, r2 in zip(ds1, ds2):
            assert r1["negative"] == r2["negative"]

    def test_none_true_author_skipped(self):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        problems = [
            {"problem_id": "p1", "unknown_text": "u1",
             "candidates": {"c1": "t1", "c2": "t2"}, "true_author": None},
            {"problem_id": "p2", "unknown_text": "u2",
             "candidates": {"c1": "t3", "c2": "t4"}, "true_author": "c1"},
        ]
        ds = PAN19Datamodule._convert_to_triplets(problems, split=None)
        assert len(ds) == 1 and ds[0]["query"] == "u2"


# ---------------------------------------------------------------------------
# Tests: tokenize
# ---------------------------------------------------------------------------


class TestTokenizeShapes:
    def test_output_keys(self, pan19_cfg):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        dm = PAN19Datamodule(
            cfg=pan19_cfg, processed_ds_dir="/tmp/test_pan19_tok", num_proc=1
        )
        result = dm.tokenize(
            {"query": ["q"], "positive": ["p"], "negative": ["n"]}, indices=[0]
        )
        required = {
            "input_ids", "attention_mask",
            "pos_input_ids", "pos_attention_mask",
            "neg_input_ids", "neg_attention_mask",
            "target_indices", "index",
        }
        assert required.issubset(set(result.keys()))

    def test_target_indices_binary(self, pan19_cfg):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        dm = PAN19Datamodule(
            cfg=pan19_cfg, processed_ds_dir="/tmp/test_pan19_tok", num_proc=1
        )
        result = dm.tokenize(
            {"query": ["q1", "q2"], "positive": ["p1", "p2"], "negative": ["n1", "n2"]},
            indices=[0, 1],
        )
        assert result["target_indices"] == [[-1], [-1]]


# ---------------------------------------------------------------------------
# Tests: full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_dataloader_keys(self, pan19_cfg, synthetic_pan19_zip, tmp_path):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        dm = PAN19Datamodule(
            cfg=pan19_cfg,
            processed_ds_dir=str(tmp_path / "proc"),
            num_proc=1,
            pan19_zip=synthetic_pan19_zip,
            split=None,
        )
        dm.setup("test")
        batch = next(iter(dm.test_dataloader()))
        for key in ("input_ids", "attention_mask",
                    "pos_input_ids", "pos_attention_mask",
                    "neg_input_ids", "neg_attention_mask"):
            assert key in batch

    def test_cache_reused(self, pan19_cfg, synthetic_pan19_zip, tmp_path):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        proc = str(tmp_path / "proc")
        dm1 = PAN19Datamodule(
            cfg=pan19_cfg, processed_ds_dir=proc, num_proc=1,
            pan19_zip=synthetic_pan19_zip, split=None,
        )
        dm1.setup("test")
        dm2 = PAN19Datamodule(
            cfg=pan19_cfg, processed_ds_dir=proc, num_proc=1,
            pan19_zip=None, split=None,  # no zip needed — cache exists
        )
        dm2.setup("test")
        assert "input_ids" in next(iter(dm2.test_dataloader()))

    def test_no_source_raises(self, pan19_cfg, tmp_path):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        dm = PAN19Datamodule(
            cfg=pan19_cfg,
            processed_ds_dir=str(tmp_path / "empty"),
            num_proc=1,
            pan19_zip=None,
            pan19_root=None,
        )
        with pytest.raises(ValueError, match="PAN19_ZIP|PAN19_ROOT|pan19_zip"):
            dm.setup("test")

    def test_dev_test_pipeline(self, pan19_cfg, synthetic_pan19_zip, tmp_path):
        from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
        dm_dev = PAN19Datamodule(
            cfg=pan19_cfg, processed_ds_dir=str(tmp_path / "proc"),
            num_proc=1, pan19_zip=synthetic_pan19_zip, split="dev",
        )
        dm_dev.setup("test")
        dm_test = PAN19Datamodule(
            cfg=pan19_cfg, processed_ds_dir=str(tmp_path / "proc"),
            num_proc=1, pan19_zip=synthetic_pan19_zip, split="test",
        )
        dm_test.setup("test")
        n_dev = len(dm_dev.test_ds)
        n_test = len(dm_test.test_ds)
        assert n_dev > 0 and n_test > 0
        assert n_dev + n_test == 6  # total valid queries across 2 English problems
