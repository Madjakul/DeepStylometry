# tests/experiments/test_patch_interactions_uses_trained_pool.py
"""Sanity-check that patch_interactions.py uses the trained PatchInteraction
from the loaded checkpoint, not a randomly-initialised fresh instance.

Requires the learned-PLI checkpoint at CHECKPOINT_PATH.  Skip cleanly when the
file is absent so CI does not break on machines without the artefact.

Run::

    python -m pytest tests/experiments/test_patch_interactions_uses_trained_pool.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

CHECKPOINT_PATH = Path(
    "tmp/answerdotai-modernbert-base__halvest__pooling-pli-learned-n3__skip_list-true/last.ckpt"
)
CONFIG_PATH = Path("configs/test_pli_learned.yml")

_checkpoint_missing = not CHECKPOINT_PATH.exists() or not CONFIG_PATH.exists()


@pytest.mark.skipif(
    _checkpoint_missing,
    reason=(
        f"Learned-PLI checkpoint or config not found "
        f"({CHECKPOINT_PATH} / {CONFIG_PATH}).  "
        "Skipping requires-checkpoint test."
    ),
)
class TestTrainedPoolIsUsed:
    """The trained PatchInteraction from the loaded model must differ from a
    fresh randomly-initialised one."""

    @pytest.fixture(scope="class")
    def model_and_cfg(self):
        from deep_stylometry.modules import DeepStylometry
        from deep_stylometry.utils.configs import BaseConfig

        cfg = BaseConfig.from_yaml(str(CONFIG_PATH))
        model = DeepStylometry.load_from_checkpoint(str(CHECKPOINT_PATH), cfg=cfg)
        model.eval()
        model.to(torch.device("cpu"))
        return model, cfg

    def test_pool_is_patch_interaction(self, model_and_cfg):
        """model.contrastive_loss.pool must be a PatchInteraction instance."""
        from deep_stylometry.modules.patch_interaction import PatchInteraction

        model, _ = model_and_cfg
        trained_pool = getattr(model.contrastive_loss, "pool", None)
        assert isinstance(trained_pool, PatchInteraction), (
            f"Expected PatchInteraction, got {type(trained_pool)}"
        )

    def test_predictor_is_not_none(self, model_and_cfg):
        """The trained pool must have a non-None predictor (learned boundary FFN)."""
        model, _ = model_and_cfg
        trained_pli = model.contrastive_loss.pool
        assert trained_pli.predictor is not None, (
            "trained_pli.predictor is None — checkpoint may not be a learned-PLI run."
        )

    def test_fresh_and_trained_produce_different_patches(self, model_and_cfg):
        """A fresh PatchInteraction must produce different patch_ids than the
        trained one.  If they happen to agree everywhere, the random FFN init
        accidentally matches the trained weights — investigate."""
        from deep_stylometry.modules.patch_interaction import PatchInteraction
        from transformers import AutoTokenizer

        model, cfg = model_and_cfg
        device = torch.device("cpu")

        trained_pli = model.contrastive_loss.pool.eval()
        fresh_pli = PatchInteraction(cfg).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)
        sample_text = (
            "Authorship attribution is the task of identifying who wrote a given "
            "text based on stylistic features such as vocabulary, syntax, and "
            "punctuation patterns."
        )
        enc = tokenizer(
            sample_text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.data.max_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.no_grad():
            embs = model(input_ids, attention_mask)

        with torch.no_grad():
            trained_ids, _ = trained_pli._compute_patches(
                embs, attention_mask, input_ids, step=None, training=False
            )
            fresh_ids, _ = fresh_pli._compute_patches(
                embs, attention_mask, input_ids, step=None, training=False
            )

        assert not torch.equal(trained_ids, fresh_ids), (
            "trained_pli and fresh_pli produced identical patch_ids.  "
            "Either the fix is not working (pli is still a fresh instance) "
            "or the random FFN init accidentally matches the trained weights "
            "— investigate."
        )
