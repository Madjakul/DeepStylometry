# tests/experiments/test_patch_interactions_pan19.py
"""Tests for the PAN19 support added to patch_interactions.py.

Neither test requires GPU, real HuggingFace downloads, or network access.
All external I/O is monkey-patched with in-memory stubs.

Run::

    python -m pytest tests/experiments/test_patch_interactions_pan19.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import datasets
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Stubs shared by both tests
# ---------------------------------------------------------------------------

_TRIPLETS = datasets.Dataset.from_dict(
    {
        "query":    ["The cat sat on the mat.", "A quick brown fox.", "Hello world."],
        "positive": ["The feline rested on the rug.", "A fast tawny fox.", "Hi earth."],
        "negative": ["The dog barked loudly.", "The slow white rabbit.", "Goodbye moon."],
    }
)


class _StubModel(nn.Module):
    """Minimal model stub: returns random (batch, seq, HIDDEN) embeddings."""

    HIDDEN = 64

    def __init__(self) -> None:
        super().__init__()
        # nn.Parameter so Lightning's load_from_checkpoint can instantiate us.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        # Expose contrastive_loss.pool = None so patch_interactions.py can call
        # getattr(model.contrastive_loss, "pool", None) without raising.
        # pool=None means the fallback warning branch fires (correct for a stub
        # that does not represent a real PLI checkpoint).
        self.contrastive_loss = SimpleNamespace(pool=None)

    def eval(self) -> "_StubModel":
        return self

    def to(self, *args: Any, **kwargs: Any) -> "_StubModel":
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        return torch.randn(b, s, self.HIDDEN)


class _TensorDict(dict):
    """dict subclass that silently accepts .to(device) like BatchEncoding does."""

    def to(self, device: Any) -> "_TensorDict":
        return self


class _StubTokenizer:
    """Tokenizer stub: returns fixed 6-token sequences."""

    _IDS = [1, 100, 101, 102, 103, 2]

    def __call__(
        self,
        text: Any,
        return_tensors: Optional[str] = None,
        truncation: bool = False,
        max_length: int = 512,
        **kwargs: Any,
    ) -> _TensorDict:
        ids = torch.tensor([self._IDS])
        mask = torch.ones_like(ids)
        return _TensorDict({"input_ids": ids, "attention_mask": mask})

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [f"tok{i}" for i in ids]


def _make_stub_cfg() -> Any:
    """Return a minimal BaseConfig suitable for ngram PatchInteraction."""
    from deep_stylometry.utils.configs import BaseConfig

    cfg = BaseConfig()
    cfg.model.base_checkpoint = "unused/model"
    cfg.model.pooling_method = "pli"
    cfg.model.patch_method = "ngram"
    cfg.model.patch_size = 2
    cfg.model.patch_compression = "mean"
    cfg.model.skip_list = False
    cfg.model.lm_hidden_size = _StubModel.HIDDEN
    cfg.model.expansion_ratio = 1
    cfg.model.patch_cross_attn_dim = 16
    cfg.model.patch_cross_attn_heads = 2
    cfg.model.patch_lambda = 0.1
    cfg.model.gumbel_tau_init = 1.0
    cfg.model.gumbel_tau_final = 0.1
    cfg.model.gumbel_anneal_steps = 100
    cfg.data.max_length = 512
    cfg.data.batch_size = 4
    cfg.data.tokenizer_name = "unused/model"
    cfg.train.tau = 0.05
    cfg.train.loss = "info_nce"
    cfg.train.gather = False
    cfg.train.precision = "32"
    return cfg


# ---------------------------------------------------------------------------
# Test 1: _load_pan19_triplets (monkey-patched PAN19Datamodule)
# ---------------------------------------------------------------------------

class TestLoadPan19Triplets:
    """_load_pan19_triplets returns correct triplets when PAN19Datamodule is patched."""

    def test_returns_expected_triplets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Helper returns the patched dataset with correct order and content."""
        from deep_stylometry.utils.data import pan19_datamodule
        from deep_stylometry.experiments.patch_interactions import _load_pan19_triplets

        # Patch _parse_problems_from_zip to return dummy problems (content irrelevant
        # because _convert_to_triplets is also patched).
        monkeypatch.setattr(
            pan19_datamodule.PAN19Datamodule,
            "_parse_problems_from_zip",
            staticmethod(lambda *a, **kw: [{"dummy": True}] * 3),
        )
        # Patch _convert_to_triplets to return our fixed dataset.
        monkeypatch.setattr(
            pan19_datamodule.PAN19Datamodule,
            "_convert_to_triplets",
            staticmethod(lambda *a, **kw: _TRIPLETS),
        )

        cfg = _make_stub_cfg()
        result = _load_pan19_triplets(
            cfg=cfg,
            n_samples=3,
            seed=42,
            language="en",
            pan19_zip="/fake/pan19.zip",
        )

        assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
        assert result[0]["query"] == _TRIPLETS[0]["query"]
        assert result[1]["positive"] == _TRIPLETS[1]["positive"]
        assert result[2]["negative"] == _TRIPLETS[2]["negative"]

    def test_n_samples_limits_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Requesting fewer than available samples truncates the output."""
        from deep_stylometry.utils.data import pan19_datamodule
        from deep_stylometry.experiments.patch_interactions import _load_pan19_triplets

        monkeypatch.setattr(
            pan19_datamodule.PAN19Datamodule,
            "_parse_problems_from_zip",
            staticmethod(lambda *a, **kw: []),
        )
        monkeypatch.setattr(
            pan19_datamodule.PAN19Datamodule,
            "_convert_to_triplets",
            staticmethod(lambda *a, **kw: _TRIPLETS),
        )

        cfg = _make_stub_cfg()
        result = _load_pan19_triplets(
            cfg=cfg,
            n_samples=2,
            seed=0,
            pan19_zip="/fake/pan19.zip",
        )
        assert len(result) == 2

    def test_raises_without_zip(self) -> None:
        """Raises ValueError when neither pan19_zip nor PAN19_ZIP env var is set."""
        from deep_stylometry.experiments.patch_interactions import _load_pan19_triplets
        import os

        # Ensure PAN19_ZIP is not set in the environment.
        env_backup = os.environ.pop("PAN19_ZIP", None)
        try:
            cfg = _make_stub_cfg()
            with pytest.raises(ValueError, match="PAN19_ZIP"):
                _load_pan19_triplets(cfg=cfg, n_samples=10, pan19_zip=None)
        finally:
            if env_backup is not None:
                os.environ["PAN19_ZIP"] = env_backup


# ---------------------------------------------------------------------------
# Test 2: smoke test — full main() with mocked model, tokenizer, and data
# ---------------------------------------------------------------------------

class TestSmokePan19:
    """main() runs end-to-end on CPU with all external dependencies mocked."""

    def test_html_is_written_with_expected_content(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """HTML file is written and contains 'PAN19' and 'Triplet accuracy'."""
        import deep_stylometry.experiments.patch_interactions as pi_mod
        from deep_stylometry.utils.configs import BaseConfig
        from deep_stylometry.modules.patch_interaction import PatchInteraction

        output_html = str(tmp_path / "test_out.html")
        cfg = _make_stub_cfg()

        # --- Patch BaseConfig.from_yaml ---
        monkeypatch.setattr(
            BaseConfig, "from_yaml", classmethod(lambda cls, path: cfg)
        )

        # --- Patch DeepStylometry.load_from_checkpoint ---
        from deep_stylometry.modules import DeepStylometry
        stub_model = _StubModel()
        monkeypatch.setattr(
            DeepStylometry,
            "load_from_checkpoint",
            classmethod(lambda cls, ckpt, **kw: stub_model),
        )

        # --- Patch AutoTokenizer.from_pretrained ---
        import transformers
        stub_tok = _StubTokenizer()
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            staticmethod(lambda name, **kw: stub_tok),
        )

        # --- Patch _load_pan19_triplets in the patch_interactions module ---
        monkeypatch.setattr(
            pi_mod,
            "_load_pan19_triplets",
            lambda *a, **kw: _TRIPLETS,
        )

        # --- Run main() with PAN19 args via sys.argv ---
        argv_backup = sys.argv[:]
        sys.argv = [
            "patch_interactions",
            "--config_path", "fake.yml",
            "--checkpoint_path", "fake.ckpt",
            "--dataset", "pan19",
            "--n_samples", "3",
            "--n_viz", "2",
            "--output_html", output_html,
            "--seed", "42",
        ]
        try:
            pi_mod.main()
        finally:
            sys.argv = argv_backup

        assert Path(output_html).exists(), "HTML file was not written."
        html_content = Path(output_html).read_text(encoding="utf-8")
        assert "PAN19" in html_content, "'PAN19' not found in HTML output."
        assert "Triplet accuracy" in html_content, (
            "'Triplet accuracy' not found in HTML output."
        )

    def test_default_output_filename_pan19(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Default output filename is 'interactions_patch_pan19.html' for pan19."""
        import deep_stylometry.experiments.patch_interactions as pi_mod
        from deep_stylometry.utils.configs import BaseConfig
        from deep_stylometry.modules import DeepStylometry
        import transformers

        cfg = _make_stub_cfg()
        monkeypatch.setattr(BaseConfig, "from_yaml", classmethod(lambda cls, p: cfg))
        monkeypatch.setattr(
            DeepStylometry,
            "load_from_checkpoint",
            classmethod(lambda cls, ckpt, **kw: _StubModel()),
        )
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            staticmethod(lambda n, **kw: _StubTokenizer()),
        )
        monkeypatch.setattr(pi_mod, "_load_pan19_triplets", lambda *a, **kw: _TRIPLETS)

        # Change working directory so default output is written inside tmp_path.
        monkeypatch.chdir(tmp_path)

        argv_backup = sys.argv[:]
        sys.argv = [
            "patch_interactions",
            "--config_path", "fake.yml",
            "--checkpoint_path", "fake.ckpt",
            "--dataset", "pan19",
            "--n_samples", "3",
            "--n_viz", "1",
        ]
        try:
            pi_mod.main()
        finally:
            sys.argv = argv_backup

        assert (tmp_path / "interactions_patch_pan19.html").exists(), (
            "Default PAN19 output file not created."
        )
