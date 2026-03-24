# tests/test_se_datamodule.py
"""Tests for StyleEmbeddingDatamodule.

The raw HuggingFace dataset is cached locally so these tests load a small
slice without network access.
"""

import pytest
import torch
from datasets import load_dataset


class TestStyleEmbeddingTripletConversion:
    def test_triplet_format_label1(self):
        """When label==1: author_A == author_U1 → U1 is the positive."""
        from deep_stylometry.utils.data.se_datamodule import StyleEmbeddingDatamodule

        # Build a tiny synthetic dataset
        import datasets as hf_datasets

        rows = [
            {
                "Anchor (A)": "text A",
                "Utterance 1 (U1)": "same author as A",
                "Utterance 2 (U2)": "different author",
                "Same Author Label": 1,
            },
            {
                "Anchor (A)": "another A",
                "Utterance 1 (U1)": "different here",
                "Utterance 2 (U2)": "same here",
                "Same Author Label": 0,
            },
        ]
        raw = hf_datasets.Dataset.from_list(rows)
        triplets = StyleEmbeddingDatamodule._convert_to_triplets(raw)

        assert triplets[0]["query"] == "text A"
        assert triplets[0]["positive"] == "same author as A"
        assert triplets[0]["negative"] == "different author"

        assert triplets[1]["query"] == "another A"
        assert triplets[1]["positive"] == "same here"
        assert triplets[1]["negative"] == "different here"

    def test_all_rows_converted(self):
        """Number of triplets matches number of raw examples."""
        from deep_stylometry.utils.data.se_datamodule import StyleEmbeddingDatamodule
        import datasets as hf_datasets

        n = 10
        rows = [
            {
                "Anchor (A)": f"A_{i}",
                "Utterance 1 (U1)": f"U1_{i}",
                "Utterance 2 (U2)": f"U2_{i}",
                "Same Author Label": i % 2,
            }
            for i in range(n)
        ]
        raw = hf_datasets.Dataset.from_list(rows)
        triplets = StyleEmbeddingDatamodule._convert_to_triplets(raw)
        assert len(triplets) == n


class TestStyleEmbeddingDatamoduleLoad:
    """Integration test against the cached HuggingFace dataset."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_cache(self):
        """Skip if the cached dataset isn't available."""
        try:
            load_dataset("AnnaWegmann/StyleEmbeddingData", split="test[:5]")
        except Exception:
            pytest.skip("StyleEmbeddingData not available in cache")

    def test_datamodule_loads(self, dummy_cfg, tmp_path):
        from deep_stylometry.utils.data.se_datamodule import StyleEmbeddingDatamodule

        dummy_cfg.data.batch_size = 4
        dm = StyleEmbeddingDatamodule(
            cfg=dummy_cfg,
            processed_ds_dir=str(tmp_path),
            num_proc=1,
        )
        dm.setup("test")
        loader = dm.test_dataloader()
        batch = next(iter(loader))

        for key in ("input_ids", "attention_mask",
                    "pos_input_ids", "pos_attention_mask",
                    "neg_input_ids", "neg_attention_mask"):
            assert key in batch, f"Missing key: {key}"

    def test_batch_shapes(self, dummy_cfg, tmp_path):
        from deep_stylometry.utils.data.se_datamodule import StyleEmbeddingDatamodule

        dummy_cfg.data.batch_size = 8
        dm = StyleEmbeddingDatamodule(
            cfg=dummy_cfg,
            processed_ds_dir=str(tmp_path),
            num_proc=1,
        )
        dm.setup("test")
        loader = dm.test_dataloader()
        batch = next(iter(loader))

        B = batch["input_ids"].shape[0]
        assert batch["input_ids"].shape[0] == B
        assert batch["pos_input_ids"].shape[0] == B
        assert batch["neg_input_ids"].shape[0] == B
        # Seq dims can differ between query/pos/neg (dynamic padding)
        assert batch["attention_mask"].shape[0] == B
        assert batch["pos_attention_mask"].shape[0] == B
