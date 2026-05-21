#!/usr/bin/env python3
# deep_stylometry/experiments/semantic_decorrelation.py
"""Semantic vs. Stylistic Cosine Similarity Decorrelation Analysis.

Three-phase analysis:
  Phase 1 — Fine-tuned model embeddings collected via trainer.test() +
             DecorrelationCallback.
  Phase 2 — E5 embeddings collected via trainer.predict().
  Phase 3 — Pearson / Spearman correlation + scatter plot.

Run::

    python -m deep_stylometry.experiments.semantic_decorrelation \\
        --checkpoint /path/to/last.ckpt \\
        --config /path/to/config.yml \\
        --processed-ds-dir $WORK_DIR/Datasets/deep-stylometry/answerdotai-modernbert-base/no-padding/ \\
        --subset base-4 \\
        --sample-size 2000 \\
        --batch-size 64 \\
        --output-dir ./figures \\
        --seed 42 \\
        --device cuda \\
        --num-proc 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import datasets as hf_datasets
import lightning as L
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from deep_stylometry.modules.modeling_deep_stylometry import DeepStylometry
from deep_stylometry.utils import train_utils
from deep_stylometry.utils.configs.base_config import BaseConfig
from deep_stylometry.utils.helpers import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Shared pooling utility


def mean_pool(
    embs: torch.Tensor,  # (batch, seq, hidden)
    mask: torch.Tensor,  # (batch, seq)
) -> torch.Tensor:
    """Masked mean pooling over the sequence dimension.

    Args:
        embs: Per-token embeddings of shape ``(batch, seq, hidden)``.
        mask: Binary attention mask of shape ``(batch, seq)``. Positions
            with value 0 are excluded from the average.

    Returns:
        Pooled embeddings of shape ``(batch, hidden)``.
    """
    m = mask.unsqueeze(-1).float()
    return (embs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)


def _add_e5_prefix(texts: List[str]) -> List[str]:
    """Prepend the E5 query prefix to every text.

    The E5 model card (intfloat/multilingual-e5-base) states that every
    passage must be prefixed with ``"query: "`` when used for similarity
    tasks.  Omitting this prefix silently produces degraded representations.

    Args:
        texts: Raw text strings.

    Returns:
        Prefixed strings, one per input text.
    """
    return [f"query: {t}" for t in texts]


# Phase 1 — DecorrelationCallback


class DecorrelationCallback(L.Callback):
    """Collects mean-pooled fine-tuned embeddings for a random subset of
    test samples while the normal ``trainer.test()`` pipeline runs.

    Args:
        sample_size: Number of triplets to sample.
        seed: Random seed for reproducible sampling.
        raw_ds: The raw (non-tokenised) HuggingFace split that backs the
            pre-tokenised test set.  Used to access text and author metadata
            after collection.
    """

    def __init__(
        self,
        sample_size: int,
        seed: int,
        raw_ds: hf_datasets.Dataset,
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.seed = seed
        self.raw_ds = raw_ds

        # Pre-select random HuggingFace row indices.
        rng = random.Random(seed)
        n = len(raw_ds)
        sampled = rng.sample(range(n), min(sample_size, n))
        self.sampled_indices: Set[int] = set(sampled)
        logger.info(
            "DecorrelationCallback: pre-selected %d / %d indices",
            len(self.sampled_indices),
            n,
        )

        # Will be populated during test.
        self.q_embs_dict: Dict[int, torch.Tensor] = {}
        self.pos_embs_dict: Dict[int, torch.Tensor] = {}
        self.neg_embs_dict: Dict[int, torch.Tensor] = {}

    # Lightning hooks

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Reset storage at the start of each test epoch.
        self.q_embs_dict.clear()
        self.pos_embs_dict.clear()
        self.neg_embs_dict.clear()

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # batch["index"] holds the original HuggingFace row indices for the
        # items in this batch (set by EvalCollator from the "index" field
        # stored by tokenize(..., with_indices=True)).
        batch_indices: List[int] = batch["index"].cpu().tolist()

        # Find positions in the batch that belong to our sample and haven't
        # been collected yet (guard against unlikely duplicate indices).
        positions: List[Tuple[int, int]] = [
            (local_pos, hf_idx)
            for local_pos, hf_idx in enumerate(batch_indices)
            if hf_idx in self.sampled_indices and hf_idx not in self.q_embs_dict
        ]
        if not positions:
            return

        local_positions = torch.tensor([p for p, _ in positions])

        # Cast to float32 — test_step outputs may be fp16 when precision="16-mixed".
        q_embs_sel = outputs["q_embs"][local_positions].float().cpu()
        q_mask_sel = outputs["q_mask"][local_positions].cpu()
        pos_embs_sel = outputs["pos_embs"][local_positions].float().cpu()
        pos_mask_sel = outputs["pos_mask"][local_positions].cpu()
        neg_embs_sel = outputs["neg_embs"][local_positions].float().cpu()
        neg_mask_sel = outputs["neg_mask"][local_positions].cpu()

        q_pooled = mean_pool(q_embs_sel, q_mask_sel)    # (n, hidden)
        pos_pooled = mean_pool(pos_embs_sel, pos_mask_sel)
        neg_pooled = mean_pool(neg_embs_sel, neg_mask_sel)

        for j, (_, hf_idx) in enumerate(positions):
            self.q_embs_dict[hf_idx] = q_pooled[j]
            self.pos_embs_dict[hf_idx] = pos_pooled[j]
            self.neg_embs_dict[hf_idx] = neg_pooled[j]

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        n_collected = len(self.q_embs_dict)
        logger.info(
            "DecorrelationCallback: collected %d / %d requested samples",
            n_collected,
            len(self.sampled_indices),
        )
        if n_collected == 0:
            logger.warning(
                "No samples were collected. Check that batch['index'] "
                "values overlap with sampled_indices."
            )

    # Convenience accessors (used by the main script after test completes)

    @property
    def collected_indices(self) -> List[int]:
        """Sorted list of HuggingFace indices for which embeddings were collected."""
        return sorted(self.q_embs_dict.keys())

    def stacked_q(self) -> torch.Tensor:
        """Shape: (N, hidden)."""
        idxs = self.collected_indices
        return torch.stack([self.q_embs_dict[i] for i in idxs])

    def stacked_pos(self) -> torch.Tensor:
        """Shape: (N, hidden)."""
        idxs = self.collected_indices
        return torch.stack([self.pos_embs_dict[i] for i in idxs])

    def stacked_neg(self) -> torch.Tensor:
        """Shape: (N, hidden)."""
        idxs = self.collected_indices
        return torch.stack([self.neg_embs_dict[i] for i in idxs])


# Phase 2 — E5 Wrapper and DataModule


class _ListDataset(Dataset):
    """Simple dataset backed by two parallel lists of token-ID lists."""

    def __init__(
        self,
        input_ids: List[List[int]],
        attention_masks: List[List[int]],
    ) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
        }


class E5DataModule(L.LightningDataModule):
    """Tokenises a flat list of texts for E5 inference via trainer.predict().

    All texts must already carry the ``"query: "`` prefix — this class
    asserts that precondition in ``setup()``.

    Args:
        texts: Flat list of prefixed texts (queries + positives + negatives
            concatenated in that order).
        tokenizer_name: HuggingFace model identifier for the E5 tokeniser.
        batch_size: Predict batch size.
        max_length: Maximum tokenisation length.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer_name: str,
        batch_size: int,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.texts = texts
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._tokenizer: Optional[AutoTokenizer] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Guard: every text must carry the E5 prefix.
        bad = [t for t in self.texts if not t.startswith("query: ")]
        assert not bad, (
            f"E5DataModule: {len(bad)} texts are missing the 'query: ' prefix. "
            f"First offender: {bad[0]!r}"
        )

        encoded = self._tokenizer(
            self.texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        self._dataset = _ListDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate,
        )

    def _collate(self, batch: List[Dict]) -> Dict:
        return self._tokenizer.pad(
            {
                "input_ids": [b["input_ids"] for b in batch],
                "attention_mask": [b["attention_mask"] for b in batch],
            },
            padding=True,
            return_tensors="pt",
        )


class E5Wrapper(L.LightningModule):
    """Thin LightningModule wrapper around Multilingual-E5 for predict().

    Args:
        model_name: HuggingFace identifier for the E5 model.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def predict_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        # Use last_hidden_state + masked mean pooling.
        # Do NOT use pooler_output — the E5 model card explicitly states that
        # sentence embeddings must be obtained via mean pooling over the
        # last hidden states, not the [CLS] pooler projection.
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.float()  # always float32


# Index alignment verification


def verify_index_alignment(
    pretok_ds: hf_datasets.Dataset,
    raw_ds: hf_datasets.Dataset,
    tokenizer: AutoTokenizer,
    n_checks: int = 3,
) -> None:
    """Sanity-check that `pretok_ds[i]["index"]` correctly points to `raw_ds`.

    Decodes ``input_ids`` from the pre-tokenised dataset and compares with
    the raw text.  Logs a warning if the decoded text looks very different.

    Args:
        pretok_ds: Pre-tokenised split loaded from disk.
        raw_ds: Raw HuggingFace split (has text columns).
        tokenizer: Tokeniser used during pre-processing (for decode).
        n_checks: Number of samples to verify.
    """
    for i in range(min(n_checks, len(pretok_ds))):
        sample = pretok_ds[i]
        hf_idx = int(sample["index"])
        raw_row = raw_ds[hf_idx]
        decoded = tokenizer.decode(
            sample["input_ids"], skip_special_tokens=True
        ).strip()
        # The raw text may differ slightly due to tokenisation normalisation;
        # check for significant word overlap as a proxy for correctness.
        raw_words = set(raw_row["query"].lower().split())
        decoded_words = set(decoded.lower().split())
        if raw_words and decoded_words:
            overlap = len(raw_words & decoded_words) / len(raw_words | decoded_words)
            if overlap < 0.3:
                logger.warning(
                    "Index alignment check %d: low word-overlap (%.2f). "
                    "hf_idx=%d decoded=%r raw=%r",
                    i,
                    overlap,
                    hf_idx,
                    decoded[:80],
                    raw_row["query"][:80],
                )
            else:
                logger.info(
                    "Index alignment check %d OK (Jaccard=%.2f, hf_idx=%d)",
                    i,
                    overlap,
                    hf_idx,
                )


# Phase 3 — Correlation analysis and plotting


def compute_correlations(
    stylistic_sim_pos: np.ndarray,
    stylistic_sim_neg: np.ndarray,
    semantic_sim_pos: np.ndarray,
    semantic_sim_neg: np.ndarray,
) -> Dict:
    """Compute Pearson and Spearman correlations between semantic and stylistic
    cosine similarities.

    Args:
        stylistic_sim_pos: Cosine similarities (query↔positive) from the
            fine-tuned model.
        stylistic_sim_neg: Cosine similarities (query↔negative) from the
            fine-tuned model.
        semantic_sim_pos: Cosine similarities (query↔positive) from E5.
        semantic_sim_neg: Cosine similarities (query↔negative) from E5.

    Returns:
        Dict with Pearson r, p-value, Spearman rho, p-value, N, and
        per-array descriptive statistics.
    """
    sem = np.concatenate([semantic_sim_pos, semantic_sim_neg])
    sty = np.concatenate([stylistic_sim_pos, stylistic_sim_neg])

    pearson_r, pearson_p = scipy.stats.pearsonr(sem, sty)
    spearman_rho, spearman_p = scipy.stats.spearmanr(sem, sty)

    return {
        "N": int(len(sem)),
        "N_pos": int(len(stylistic_sim_pos)),
        "N_neg": int(len(stylistic_sim_neg)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "stylistic_sim_pos_mean": float(stylistic_sim_pos.mean()),
        "stylistic_sim_pos_std": float(stylistic_sim_pos.std()),
        "stylistic_sim_neg_mean": float(stylistic_sim_neg.mean()),
        "stylistic_sim_neg_std": float(stylistic_sim_neg.std()),
        "semantic_sim_pos_mean": float(semantic_sim_pos.mean()),
        "semantic_sim_pos_std": float(semantic_sim_pos.std()),
        "semantic_sim_neg_mean": float(semantic_sim_neg.mean()),
        "semantic_sim_neg_std": float(semantic_sim_neg.std()),
    }


def make_scatter_plot(
    stylistic_sim_pos: np.ndarray,
    stylistic_sim_neg: np.ndarray,
    semantic_sim_pos: np.ndarray,
    semantic_sim_neg: np.ndarray,
    pearson_r: float,
    pearson_p: float,
    output_dir: str,
    name_stem: str = "semantic_decorrelation",
) -> None:
    """Generate and save the scatter plot.

    Args:
        stylistic_sim_pos: Same-author stylistic cosine sims.
        stylistic_sim_neg: Different-author stylistic cosine sims.
        semantic_sim_pos: Same-author semantic cosine sims.
        semantic_sim_neg: Different-author semantic cosine sims.
        pearson_r: Pearson correlation coefficient (all pairs).
        pearson_p: Pearson p-value.
        output_dir: Directory to write PDF and PNG.
        name_stem: Base filename (without extension).
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        semantic_sim_pos,
        stylistic_sim_pos,
        alpha=0.3,
        color="#2166ac",
        label="Same-author (pos)",
        s=10,
    )
    ax.scatter(
        semantic_sim_neg,
        stylistic_sim_neg,
        alpha=0.3,
        color="#d6604d",
        label="Diff-author (neg)",
        s=10,
    )

    ax.set_xlabel("Semantic Similarity (Multilingual-E5)")
    ax.set_ylabel("Stylistic Similarity (Fine-tuned Model)")

    # Annotate Pearson r and p-value in upper-left corner.
    p_str = f"{pearson_p:.2e}" if pearson_p < 0.001 else f"{pearson_p:.3f}"
    ax.text(
        0.05,
        0.95,
        f"$r$ = {pearson_r:.3f}\n$p$ = {p_str}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{name_stem}.{ext}")
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)

    plt.close(fig)


# Main entry point


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Semantic vs. Stylistic Decorrelation Analysis"
    )
    p.add_argument("--checkpoint", required=True, help="Path to last.ckpt")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--processed-ds-dir",
        required=True,
        help="Root dir of pre-tokenised data (e.g. .../no-padding/)",
    )
    # Decorrelation-specific args default to None so the config YAML's
    # `decorrelation:` section fills them in; explicit CLI values always win.
    p.add_argument("--subset", default=None, help="HALvest subset (default from config)")
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument(
        "--e5-model",
        default=None,
        help="E5 model name (default from config or intfloat/multilingual-e5-base)",
    )
    p.add_argument("--e5-max-length", type=int, default=512)
    p.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional HuggingFace cache directory",
    )
    return p


def _load_decorrelation_cfg(config_path: str) -> Dict:
    """Read the raw ``decorrelation:`` section from the config YAML.

    ``BaseConfig.from_yaml`` ignores unknown top-level keys; we parse the
    YAML directly here so that the shell scripts only need to pass
    ``--config`` and ``--checkpoint``.

    Args:
        config_path: Path to the experiment YAML file.

    Returns:
        Dict of decorrelation-specific settings (empty if section absent).
    """
    import yaml

    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    return raw.get("decorrelation", {})


def main() -> None:
    """Run the full decorrelation analysis pipeline."""
    args = _build_arg_parser().parse_args()

    # Fill in None CLI args from the config's `decorrelation:` section.
    dcfg = _load_decorrelation_cfg(args.config)
    if args.subset is None:
        args.subset = dcfg.get("subset", "base-4")
    if args.sample_size is None:
        args.sample_size = int(dcfg.get("sample_size", 2000))
    if args.batch_size is None:
        args.batch_size = int(dcfg.get("batch_size", 64))
    if args.output_dir is None:
        args.output_dir = dcfg.get("output_dir", "./figures")
    if args.seed is None:
        args.seed = int(dcfg.get("seed", 42))
    if args.e5_model is None:
        args.e5_model = dcfg.get("e5_model", "intfloat/multilingual-e5-base")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config and override test subset.
    cfg = BaseConfig.from_yaml(args.config)
    cfg.data.test_subset = args.subset
    cfg.data.batch_size = args.batch_size

    # Load raw HuggingFace test split (has text + author metadata).
    logger.info("Loading raw HuggingFace test split '%s' …", args.subset)
    raw_ds: hf_datasets.Dataset = hf_datasets.load_dataset(
        "almanach/halvest-contrastive",
        name=args.subset,
        split="test",
        cache_dir=args.hf_cache_dir,
    )
    logger.info("Raw test split: %d rows, columns: %s", len(raw_ds), raw_ds.column_names)

    # Phase 1 — Fine-tuned model test pass.
    logger.info("=== Phase 1: Fine-tuned model embeddings ===")

    # Set up datamodule (loads pre-tokenised data from disk).
    dm = train_utils.setup_datamodule(
        cfg=cfg,
        processed_ds_dir=args.processed_ds_dir,
        num_proc=args.num_proc,
    )

    # Index alignment verification — check before loading the model so CUDA
    # is not yet initialised and multiprocessing in datasets works normally.
    dm.setup(stage="test")
    from deep_stylometry.utils.helpers import get_tokenizer

    tokenizer_ds = get_tokenizer(cfg.data.tokenizer_name)
    verify_index_alignment(dm.test_ds, raw_ds, tokenizer_ds, n_checks=3)

    # Load fine-tuned model from checkpoint.
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model = DeepStylometry.load_from_checkpoint(args.checkpoint, cfg=cfg)
    # Lightning's trainer.test() calls model.eval() automatically.

    dedup_callback = DecorrelationCallback(
        sample_size=args.sample_size,
        seed=args.seed,
        raw_ds=raw_ds,
    )

    # TestEvalCallback is not needed here — it would run LI full-corpus scoring
    # which is both unnecessary and OOMs on GPUs with < 80 GB VRAM.
    # The test pass is run solely to drive DecorrelationCallback.
    trainer = L.Trainer(
        accelerator=args.device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[dedup_callback],
    )

    trainer.test(model=model, datamodule=dm)

    collected = dedup_callback.collected_indices
    logger.info("Collected embeddings for %d triplets.", len(collected))
    if not collected:
        raise RuntimeError(
            "No embeddings collected. Ensure the pre-tokenised test data "
            "uses the same indexing as the raw HuggingFace split."
        )

    # Retrieve pooled fine-tuned embeddings.
    sty_q = dedup_callback.stacked_q()    # (N, H)
    sty_pos = dedup_callback.stacked_pos()
    sty_neg = dedup_callback.stacked_neg()

    # Phase 2 — E5 embeddings.
    logger.info("=== Phase 2: E5 embeddings ===")

    # Gather raw texts for the collected indices.
    query_texts = [raw_ds[i]["query"] for i in collected]
    pos_texts = [raw_ds[i]["positive"] for i in collected]
    neg_texts = [raw_ds[i]["negative"] for i in collected]

    # Prepend E5 "query: " prefix to every text.
    all_texts = _add_e5_prefix(query_texts + pos_texts + neg_texts)

    # Verify vocab_size match between E5 tokeniser and model.
    e5_tokenizer = AutoTokenizer.from_pretrained(args.e5_model)
    e5_model_cfg = AutoModel.from_pretrained(args.e5_model).config
    assert e5_tokenizer.vocab_size == e5_model_cfg.vocab_size, (
        f"E5 tokenizer vocab_size ({e5_tokenizer.vocab_size}) != "
        f"model vocab_size ({e5_model_cfg.vocab_size})"
    )
    del e5_model_cfg  # free config object

    e5_dm = E5DataModule(
        texts=all_texts,
        tokenizer_name=args.e5_model,
        batch_size=args.batch_size,
        max_length=args.e5_max_length,
    )
    e5_wrapper = E5Wrapper(model_name=args.e5_model)

    e5_trainer = L.Trainer(
        accelerator=args.device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    # trainer.predict() handles eval mode and no_grad internally.
    predict_outputs: List[torch.Tensor] = e5_trainer.predict(
        model=e5_wrapper, datamodule=e5_dm
    )
    e5_all = torch.cat(predict_outputs, dim=0).float()  # (3N, H_e5)

    N = len(collected)
    e5_q = e5_all[:N]
    e5_pos = e5_all[N : 2 * N]
    e5_neg = e5_all[2 * N : 3 * N]

    # Phase 3 — Correlation analysis.
    logger.info("=== Phase 3: Correlation analysis ===")

    sty_q_n = F.normalize(sty_q, p=2, dim=-1)
    sty_pos_n = F.normalize(sty_pos, p=2, dim=-1)
    sty_neg_n = F.normalize(sty_neg, p=2, dim=-1)
    e5_q_n = F.normalize(e5_q, p=2, dim=-1)
    e5_pos_n = F.normalize(e5_pos, p=2, dim=-1)
    e5_neg_n = F.normalize(e5_neg, p=2, dim=-1)

    stylistic_sim_pos = (sty_q_n * sty_pos_n).sum(-1).cpu().numpy()  # (N,)
    stylistic_sim_neg = (sty_q_n * sty_neg_n).sum(-1).cpu().numpy()
    semantic_sim_pos = (e5_q_n * e5_pos_n).sum(-1).cpu().numpy()
    semantic_sim_neg = (e5_q_n * e5_neg_n).sum(-1).cpu().numpy()

    stats = compute_correlations(
        stylistic_sim_pos, stylistic_sim_neg,
        semantic_sim_pos, semantic_sim_neg,
    )
    logger.info("Pearson r=%.4f (p=%.2e)", stats["pearson_r"], stats["pearson_p"])
    logger.info(
        "Spearman ρ=%.4f (p=%.2e)", stats["spearman_rho"], stats["spearman_p"]
    )

    # Save JSON sidecar.
    json_path = os.path.join(args.output_dir, "semantic_decorrelation_stats.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Saved statistics → %s", json_path)

    # Generate scatter plot.
    make_scatter_plot(
        stylistic_sim_pos=stylistic_sim_pos,
        stylistic_sim_neg=stylistic_sim_neg,
        semantic_sim_pos=semantic_sim_pos,
        semantic_sim_neg=semantic_sim_neg,
        pearson_r=stats["pearson_r"],
        pearson_p=stats["pearson_p"],
        output_dir=args.output_dir,
    )

    print(json.dumps(stats, indent=2))
    logger.info("Done.")


if __name__ == "__main__":
    main()
