# deep_stylometry/experiments/_patch_diagnostics.py
"""Diagnostic helpers for learned-PLI patch-behaviour analysis.

All public functions are called only when ``--diagnostic`` is passed to
``patch_interactions.py``.  They produce no side effects on the normal fast path.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from deep_stylometry.modules.patch_interaction import PatchInteraction
    from deep_stylometry.utils.configs import BaseConfig

logger = logging.getLogger(__name__)

INFERENCE_MODES = ["threshold_0.5", "threshold_0.3", "threshold_0.7", "gumbel"]
_THRESHOLD_MAP: Dict[str, float] = {
    "threshold_0.5": 0.5,
    "threshold_0.3": 0.3,
    "threshold_0.7": 0.7,
}


# ---------------------------------------------------------------------------
# Patch-ID helpers
# ---------------------------------------------------------------------------

def _patch_ids_from_cut_probs(
    cut_probs: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Recompute patch IDs from raw sigmoid cut probabilities using a hard threshold.

    Mirrors the inference branch of ``PatchBoundaryPredictor.forward`` exactly:
    first valid token per sequence is forced to be a boundary, padding is -1.

    Args:
        cut_probs: ``(B, S)`` raw sigmoid boundary probabilities in ``[0, 1]``.
        mask: ``(B, S)`` attention mask; 1 = real token, 0 = padding.
        threshold: A token starts a new patch when ``cut_probs[i] > threshold``.

    Returns:
        ``(B, S)`` integer patch IDs; -1 for padding positions.
    """
    first_valid = ((mask.cumsum(dim=1) == 1) & (mask > 0)).float()
    cuts = (cut_probs > threshold).float()
    cuts = torch.clamp(cuts + first_valid, max=1.0)
    cuts = cuts * mask.float()
    raw_ids = cuts.cumsum(dim=-1).long() - 1
    return raw_ids.masked_fill(mask == 0, -1)


def _patch_lengths_from_ids(
    patch_ids: torch.Tensor,
    mask: torch.Tensor,
) -> List[int]:
    """Extract per-patch lengths from a ``(1, S)`` patch-ID tensor.

    Args:
        patch_ids: ``(1, S)`` integer patch assignments; -1 for padding.
        mask: ``(1, S)`` attention mask.

    Returns:
        List of patch lengths (one value per patch) for the single sequence.
    """
    max_p = max(1, int(patch_ids.clamp(min=0).max().item()) + 1) if patch_ids.max() >= 0 else 0
    lengths: List[int] = []
    for pid in range(max_p):
        plen = int(((patch_ids[0] == pid) & (mask[0] > 0)).sum().item())
        if plen > 0:
            lengths.append(plen)
    return lengths


def compute_patches_with_diagnostics(
    pli: "PatchInteraction",
    embs: torch.Tensor,
    mask: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    inference_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute patch IDs under the requested inference rule, also returning
    the raw cut probabilities for diagnostic use.

    Always calls ``pli.predictor`` with ``training=False`` to obtain the
    deterministic raw sigmoid cut probabilities.  The threshold or Gumbel
    decision is then applied separately according to ``inference_mode``.

    Args:
        pli: Trained :class:`~deep_stylometry.modules.patch_interaction.PatchInteraction`
            with a non-``None`` predictor.
        embs: ``(B, S, H)`` contextualised token embeddings.
        mask: ``(B, S)`` attention mask.
        input_ids: Unused here; retained for API symmetry with
            :meth:`~deep_stylometry.modules.patch_interaction.PatchInteraction._compute_patches`.
        inference_mode: One of ``{"threshold_0.5", "threshold_0.3",
            "threshold_0.7", "gumbel"}``.

    Returns:
        patch_ids: ``(B, S)`` integer patch assignments; -1 for padding.
        cut_probs: ``(B, S)`` raw sigmoid boundary probabilities.

    Raises:
        ValueError: If ``pli.predictor is None`` or ``inference_mode`` is unknown.
    """
    if pli.predictor is None:
        raise ValueError(
            "compute_patches_with_diagnostics called on a non-learned "
            "PatchInteraction (predictor is None)."
        )

    # Always obtain deterministic raw cut_probs via the inference branch.
    _, cut_probs, _ = pli.predictor(embs, mask, step=None, training=False)

    if inference_mode in _THRESHOLD_MAP:
        patch_ids = _patch_ids_from_cut_probs(cut_probs, mask, _THRESHOLD_MAP[inference_mode])
    elif inference_mode == "gumbel":
        # Gumbel sampling is stochastic; the caller must seed torch for
        # reproducibility in tests.  In production diagnostic runs the
        # result varies across calls.
        patch_ids, _, _ = pli.predictor(embs, mask, step=None, training=True)
    else:
        raise ValueError(f"Unknown inference_mode: {inference_mode!r}")

    return patch_ids, cut_probs


# ---------------------------------------------------------------------------
# Population cut-probability statistics
# ---------------------------------------------------------------------------

def _ascii_histogram(
    values: List[float],
    bins: int = 50,
    bar_width: int = 60,
    range_: Tuple[float, float] = (0.0, 1.0),
) -> List[str]:
    """Build a fixed-width ASCII bar chart for a 1-D distribution.

    Args:
        values: Raw floating-point values to bin.
        bins: Number of histogram bins.
        bar_width: Maximum bar length in characters (capped per-bar).
        range_: Explicit ``(lo, hi)`` range for ``np.histogram``.

    Returns:
        One line per bin; suitable for writing to a terminal or SLURM log.
    """
    arr = np.array(values, dtype=np.float32)
    counts, edges = np.histogram(arr, bins=bins, range=range_)
    max_count = int(counts.max()) if counts.max() > 0 else 1
    lines: List[str] = []
    for i, cnt in enumerate(counts):
        midpoint = float((edges[i] + edges[i + 1]) / 2)
        bar_len = int(bar_width * cnt / max_count)
        lines.append(
            f"  {midpoint:.3f} | {'█' * bar_len:<{bar_width}s} {int(cnt):6d}"
        )
    return lines


def print_and_save_cut_prob_stats(
    all_cut_probs: List[float],
    output_path: Path,
) -> dict:
    """Compute, print to stdout, and save population statistics for cut probs.

    Args:
        all_cut_probs: Flat list of raw cut-probability values for every
            valid (non-padding) query token across all processed triplets.
        output_path: Write path for ``cut_probs_stats.json``.

    Returns:
        Stats dict that was also serialised to ``output_path``.
    """
    arr = np.array(all_cut_probs, dtype=np.float32)
    pct_keys = [5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(arr, pct_keys).tolist()

    stats = {
        "n_values": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "percentiles": {str(k): float(v) for k, v in zip(pct_keys, pct_vals)},
        "frac_above_0.3": float((arr > 0.3).mean()),
        "frac_above_0.5": float((arr > 0.5).mean()),
        "frac_above_0.7": float((arr > 0.7).mean()),
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True, default=str)
    logger.info("Saved cut_probs_stats to %s", output_path)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1 — Cut-Probability Distribution")
    print("=" * 70)
    print(f"  n_values : {stats['n_values']:,}")
    print(f"  mean     : {stats['mean']:.4f}")
    print(f"  median   : {stats['median']:.4f}")
    print(f"  std      : {stats['std']:.4f}")
    print(f"  frac>0.3 : {stats['frac_above_0.3']:.4f}")
    print(f"  frac>0.5 : {stats['frac_above_0.5']:.4f}")
    print(f"  frac>0.7 : {stats['frac_above_0.7']:.4f}")
    print("\n  Percentiles:")
    for k, v in zip(pct_keys, pct_vals):
        print(f"    p{k:2d} = {v:.4f}")
    print("\n  ASCII Histogram (cut_prob ∈ [0,1], 50 bins):")
    for line in _ascii_histogram(all_cut_probs):
        print(line)
    print()

    return stats


# ---------------------------------------------------------------------------
# Per-mode patch-length statistics
# ---------------------------------------------------------------------------

def _patch_length_stats(lengths: List[int]) -> dict:
    """Summarise a flat list of patch lengths.

    Args:
        lengths: One value per patch across all sequences.

    Returns:
        Dict with summary statistics.
    """
    if not lengths:
        return {"error": "no patches collected"}
    arr = np.array(lengths, dtype=np.float32)
    return {
        "total_patches": int(len(lengths)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "frac_len1": float((arr == 1).mean()),
        "frac_len_gt5": float((arr > 5).mean()),
        "most_common": Counter(lengths).most_common(10),
    }


def compute_and_save_patch_lengths_by_mode(
    cut_probs_records: List[Tuple[torch.Tensor, torch.Tensor]],
    gumbel_patch_ids_records: List[torch.Tensor],
    output_path: Path,
) -> dict:
    """Compute patch-length distributions under all four inference rules,
    print a compact comparison table, and save to JSON.

    Args:
        cut_probs_records: List of ``(cut_probs, mask)`` tensors (both ``(1, S)``)
            collected from each processed query, on CPU.
        gumbel_patch_ids_records: List of ``(1, S)`` patch-ID tensors from
            Gumbel inference, one per query.
        output_path: Write path for ``patch_length_by_inference_mode.json``.

    Returns:
        Dict mapping inference-mode name → stats dict.
    """
    results: Dict[str, dict] = {}

    for mode in INFERENCE_MODES:
        lengths: List[int] = []

        if mode == "gumbel":
            for g_ids in gumbel_patch_ids_records:
                lengths.extend(_patch_lengths_from_ids(g_ids, (g_ids >= 0).long()))
        else:
            threshold = _THRESHOLD_MAP[mode]
            for cut_probs, mask in cut_probs_records:
                ids = _patch_ids_from_cut_probs(cut_probs, mask, threshold)
                lengths.extend(_patch_lengths_from_ids(ids, mask))

        results[mode] = _patch_length_stats(lengths)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    logger.info("Saved patch_length_by_inference_mode to %s", output_path)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2 — Patch-Length by Inference Mode")
    print("=" * 70)
    col = f"{'Mode':<20s}  {'Mean':>7s}  {'Median':>7s}  {'frac_len1':>10s}  {'frac_len>5':>10s}"
    print(col)
    print("-" * len(col))
    for mode in INFERENCE_MODES:
        s = results[mode]
        if "error" in s:
            print(f"  {mode:<18s}  (no data)")
            continue
        print(
            f"  {mode:<18s}  {s['mean']:>7.2f}  {s['median']:>7.1f}"
            f"  {s['frac_len1']:>10.4f}  {s['frac_len_gt5']:>10.4f}"
        )
    print()

    return results


# ---------------------------------------------------------------------------
# Predictor sanity checks (run once at startup)
# ---------------------------------------------------------------------------

def run_predictor_sanity_checks(
    pli: "PatchInteraction",
    model: torch.nn.Module,
    cfg: "BaseConfig",
) -> None:
    """Run pre-loop sanity checks on the loaded predictor.

    Checks:
    1. ``pli is model.contrastive_loss.pool`` (Python identity).
    2. ``pli.predictor is not None``.
    3. Loaded parameter norm vs. a fresh randomly-initialised predictor of
       the same architecture — if norms are within 1% a WARNING is emitted
       because it suggests the weights were not deserialised.
    4. Weight dtypes of the predictor's FFN layers — mismatch across layers
       is logged as a WARNING.

    Args:
        pli: The :class:`~deep_stylometry.modules.patch_interaction.PatchInteraction`
            instance being used for analysis.
        model: Loaded ``DeepStylometry`` checkpoint.
        cfg: Config used to load the checkpoint.

    Raises:
        RuntimeError: If check 1 or 2 fails.
    """
    from deep_stylometry.modules.patch_boundary_predictor import PatchBoundaryPredictor

    # --- Check 1: Python identity ---
    pool = getattr(model.contrastive_loss, "pool", None)
    if pli is not pool:
        raise RuntimeError(
            "SANITY FAIL: pli is NOT model.contrastive_loss.pool.  "
            "The fix that reuses the trained PatchInteraction may not have "
            f"taken effect.  id(pli)={id(pli)}, id(pool)={id(pool)}"
        )
    logger.info("[SANITY 1] pli is model.contrastive_loss.pool ✓")

    # --- Check 2: predictor not None ---
    if pli.predictor is None:
        raise RuntimeError(
            "SANITY FAIL: pli.predictor is None — checkpoint is not a "
            "learned-PLI run.  Diagnostic mode requires a trained boundary predictor."
        )
    logger.info("[SANITY 2] pli.predictor is not None ✓")

    # --- Check 3: parameter norms ---
    loaded_norm = sum(p.data.norm().item() for p in pli.predictor.parameters())
    fresh = PatchBoundaryPredictor(cfg.model.lm_hidden_size, cfg)
    fresh_norm = sum(p.data.norm().item() for p in fresh.parameters())
    logger.info(
        "[SANITY 3] Predictor param_norm: loaded=%.4f  fresh_random=%.4f",
        loaded_norm, fresh_norm,
    )
    if fresh_norm > 0 and abs(loaded_norm - fresh_norm) / fresh_norm < 0.01:
        logger.warning(
            "[SANITY 3] WARNING — loaded and fresh-random norms within 1%% "
            "(loaded=%.4f, fresh=%.4f).  Loaded weights may not have been "
            "deserialised from the checkpoint; verify the .ckpt contains "
            "predictor parameters.",
            loaded_norm, fresh_norm,
        )
    else:
        logger.info(
            "[SANITY 3] Norm difference is substantial (>1%%) — loaded weights "
            "look distinct from random init ✓"
        )

    # --- Check 4: weight dtypes ---
    dtype0 = pli.predictor.ffn[0].weight.dtype
    dtype2 = pli.predictor.ffn[2].weight.dtype
    logger.info(
        "[SANITY 4] predictor.ffn[0].weight.dtype=%s  ffn[2].weight.dtype=%s",
        dtype0, dtype2,
    )
    if dtype0 != dtype2:
        logger.warning(
            "[SANITY 4] WARNING — FFN layer dtype mismatch: ffn[0]=%s vs "
            "ffn[2]=%s.  Unexpected; investigate whether mixed-precision "
            "serialisation caused the divergence.",
            dtype0, dtype2,
        )
    # REVIEW: PatchBoundaryPredictor.forward casts token_embs to ffn[0].weight.dtype.
    # Under 16-mixed training, activations arrive as fp16 but weights are kept
    # fp32 (PyTorch AMP default).  At CPU inference both should be fp32, but if
    # the checkpoint was saved mid-training under AMP it may have fp16 weight
    # copies.  The dtype check above surfaces this.
    logger.info("[SANITY] All predictor sanity checks complete.")


# ---------------------------------------------------------------------------
# forward() vs _compute_patches() cross-check (first triplet only)
# ---------------------------------------------------------------------------

def run_forward_cross_check(
    pli: "PatchInteraction",
    q_embs: torch.Tensor,
    q_mask: torch.Tensor,
    q_input_ids: torch.Tensor,
    k_embs: torch.Tensor,
    k_mask: torch.Tensor,
    k_input_ids: torch.Tensor,
) -> None:
    """Assert that ``pli.forward`` and ``pli._compute_patches`` produce
    identical cut probabilities for the query side of the first triplet.

    ``pli.forward()`` stores the query cut_probs in ``_last_cut_probs`` after
    each call.  If the two paths diverge (e.g. due to dtype casting or
    in-place normalisation that exists in forward but not in the direct call),
    it would mean the script's patch statistics differ from those seen by
    ``TestEvalCallback``.

    Args:
        pli: Trained PatchInteraction in eval mode.
        q_embs, q_mask, q_input_ids: Query tensors from the first triplet.
        k_embs, k_mask, k_input_ids: Key tensors from the first triplet.
    """
    with torch.no_grad():
        direct_ids, direct_cut_probs = pli._compute_patches(
            q_embs, q_mask, q_input_ids, step=None, training=False
        )
        _ = pli(
            query_embs=q_embs,
            key_embs=k_embs,
            q_mask=q_mask,
            k_mask=k_mask,
            q_input_ids=q_input_ids,
            k_input_ids=k_input_ids,
            step=None,
        )
        forward_cut_probs = pli._last_cut_probs  # written by forward()

    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC 4 — forward() vs _compute_patches() cross-check")
    logger.info("=" * 70)

    if direct_cut_probs is None:
        logger.info("[CROSS-CHECK] Non-learned patching — cut_probs comparison N/A.")
        return

    if forward_cut_probs is None:
        logger.warning(
            "[CROSS-CHECK] pli._last_cut_probs is None after forward() call.  "
            "Something unexpected happened in the forward path."
        )
        return

    if torch.equal(direct_cut_probs, forward_cut_probs):
        logger.info(
            "[CROSS-CHECK] forward() and _compute_patches() produce identical "
            "cut_probs for the query ✓"
        )
    else:
        diff = (direct_cut_probs - forward_cut_probs).abs()
        logger.error(
            "[CROSS-CHECK] MISMATCH — forward() and _compute_patches() give "
            "different cut_probs.  max_abs_diff=%.6f  mean_abs_diff=%.6f  "
            "n_positions_differ=%d.  This means the script's patch boundaries "
            "differ from TestEvalCallback.  Investigate dtype handling in the "
            "forward path.",
            diff.max().item(), diff.mean().item(), (diff > 0).sum().item(),
        )

    # Derive patch_ids from both sets of cut_probs and compare
    if direct_cut_probs is not None and forward_cut_probs is not None:
        fwd_ids = _patch_ids_from_cut_probs(forward_cut_probs, q_mask, 0.5)
        if torch.equal(direct_ids, fwd_ids):
            logger.info("[CROSS-CHECK] Derived patch_ids agree ✓")
        else:
            n_diff = (direct_ids != fwd_ids).sum().item()
            logger.error(
                "[CROSS-CHECK] Derived patch_ids differ at %d positions — "
                "a cut_prob diff straddled the 0.5 threshold.",
                n_diff,
            )


# ---------------------------------------------------------------------------
# Per-token cut_prob HTML heatmap (opt-in, first N diagnostic pairs)
# ---------------------------------------------------------------------------

def cut_prob_heatmap_html(
    tokens: List[str],
    cut_probs_1d: torch.Tensor,
    mask_1d: torch.Tensor,
    pair_idx: str,
) -> str:
    """Return an HTML block showing per-token cut_prob as a colour-coded heatmap.

    Cells transition from white (prob ≈ 0) to red (prob ≈ 1).  A small
    triangle marker (▲) is prepended to tokens where cut_prob > 0.5 to make
    the default-threshold boundary positions immediately visible.

    Args:
        tokens: Decoded token strings for the query sequence.
        cut_probs_1d: ``(S,)`` raw sigmoid boundary probabilities.
        mask_1d: ``(S,)`` attention mask.
        pair_idx: Identifier used in the header label.

    Returns:
        Self-contained HTML fragment; insert above the patch visualisation.
    """
    parts: List[str] = []
    for tok, m, cp in zip(tokens, mask_1d.tolist(), cut_probs_1d.cpu().tolist()):
        if m == 0:
            continue
        intensity = int(255 * (1.0 - float(cp)))  # high prob → red (low intensity)
        bg = f"rgb(255,{intensity},{intensity})"
        marker = "▲" if cp > 0.5 else ""
        tok_clean = (
            tok.replace("Ġ", " ")
               .replace("<", "&lt;")
               .replace(">", "&gt;")
        )
        parts.append(
            f"<span style='background:{bg};padding:1px 3px;border-radius:2px;"
            f"font-size:12px;margin:1px;display:inline-block;white-space:pre;' "
            f"title='cut_prob={cp:.4f}'>{marker}{tok_clean}</span>"
        )

    return (
        "<div style='margin-bottom:8px;border:1px dashed #ccc;padding:8px;"
        "border-radius:4px;'>"
        "<p style='font-size:11px;color:#888;margin:0 0 4px;"
        f"font-family:monospace;'>[{pair_idx}] Cut-probability heatmap "
        "(▲ = cut_prob &gt; 0.5 → boundary under default threshold; "
        "red = high, white = low)</p>"
        "<div style='line-height:2.0;font-family:monospace;font-size:13px;'>"
        f"{''.join(parts)}</div></div>\n"
    )
