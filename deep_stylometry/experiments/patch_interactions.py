# deep_stylometry/experiments/patch_interactions.py
"""Patch-level interaction visualiser.

Analogous to ``token_interactions.py`` but operates at the patch level.
Renders an interactive HTML with patch spans, MaxSim alignments, and
per-patch score contributions.  Also outputs summary statistics:
  - Average patch length distribution
  - Patch boundary positions relative to POS tags (via spaCy if available)
  - Most common patch lengths

Supports two datasets:
  - ``halvest`` (default): HALvest-Contrastive, loaded directly from HuggingFace.
  - ``pan19``: PAN 2019 CDAA, loaded via :func:`_load_pan19_triplets`.

PAN19 note: the learned-PLI policy applied to PAN19's ~512-token
post-truncation unknowns may show different mean patch lengths than on
HALvest base-k subsets.  Run with several ``--seed`` values and average for a
robust estimate (e.g. seeds 42, 0, 1, 2, 3).

Usage
-----
# HALvest (default)
python -m deep_stylometry.experiments.patch_interactions \\
    --config_path configs/test_pli_learned.yml \\
    --checkpoint_path tmp/.../last.ckpt \\
    --subset base-2 --n_samples 200 --n_viz 50

# With diagnostics
python -m deep_stylometry.experiments.patch_interactions \\
    --config_path configs/test_pli_learned.yml \\
    --checkpoint_path tmp/.../last.ckpt \\
    --subset base-2 --n_samples 200 --diagnostic --inference_mode threshold_0.5

# PAN19
python -m deep_stylometry.experiments.patch_interactions \\
    --config_path configs/test_pli_learned.yml \\
    --checkpoint_path tmp/.../last.ckpt \\
    --dataset pan19 --n_samples 200 --seed 42 \\
    --output_html interactions_patch_pan19.html
"""

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Number of pairs for which we emit the per-token cut_prob heatmap in
# diagnostic mode (in addition to the normal patch visualisation).
_N_DIAG_VIZ = 5


# ---------------------------------------------------------------------------
# Core analysis helpers (unchanged from original)
# ---------------------------------------------------------------------------

def compute_patch_alignments(
    q_patch_embs: torch.Tensor,   # (1, Pq, H)
    k_patch_embs: torch.Tensor,   # (1, Pk, H)
    q_patch_mask: torch.Tensor,   # (1, Pq)
    k_patch_mask: torch.Tensor,   # (1, Pk)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-query-patch MaxSim scores and alignment indices.

    Returns
    -------
    scores_1d: (Pq,) per-patch MaxSim score
    align_1d:  (Pq,) matched key-patch index for each query patch
    patch_score: scalar total score
    """
    q_norm = F.normalize(q_patch_embs, p=2, dim=-1)
    k_norm = F.normalize(k_patch_embs, p=2, dim=-1)

    sim = torch.einsum("bsh,bth->bst", q_norm, k_norm)  # (1, Pq, Pk)
    # k_patch_mask is (1, Pk); unsqueeze to (1, 1, Pk) to broadcast with (1, Pq, Pk)
    mask_inv = (1.0 - k_patch_mask.float()).unsqueeze(1)  # (1, 1, Pk)
    sim = sim + mask_inv * (-10000.0)

    max_result = sim[0].max(dim=-1)  # (Pq,)
    scores_1d = max_result.values.squeeze(0) if max_result.values.dim() > 1 else max_result.values
    align_1d = max_result.indices.squeeze(0) if max_result.indices.dim() > 1 else max_result.indices

    scores_1d_masked = scores_1d * q_patch_mask[0].float()
    patch_score = scores_1d_masked.sum()

    return scores_1d.cpu(), align_1d.cpu(), patch_score


def get_patch_spans(
    token_strings: List[str],
    patch_ids: torch.Tensor,  # (S,)
    mask: torch.Tensor,       # (S,)
) -> List[List[int]]:
    """Return a list of token-index lists, one per patch.

    Returns
    -------
    spans: List of token index lists ordered by patch_id
    """
    spans: dict = {}
    for i, (tok, m, pid) in enumerate(zip(token_strings,
                                          mask.tolist(),
                                          patch_ids.tolist())):
        if m == 0 or pid < 0:
            continue
        spans.setdefault(pid, []).append(i)
    return [spans[pid] for pid in sorted(spans.keys())]


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

PATCH_COLORS = [
    "100,149,237",   # Cornflower blue
    "255,160,122",   # Light salmon
    "152,251,152",   # Pale green
    "221,160,221",   # Plum
    "255,218,185",   # Peach
    "176,196,222",   # Light steel blue
    "255,228,181",   # Moccasin
    "173,216,230",   # Light blue
]


def _patch_span_html(
    token: str,
    patch_idx: int,
    token_idx: int,
    score: float,
    align_target: int,
    side: str,
    pair_idx: str,
    min_s: float,
    max_s: float,
    cut_prob: Optional[float] = None,
) -> str:
    """Render a single token span.

    Args:
        token: Raw token string (Ġ prefix preserved; stripped for display).
        patch_idx: Patch index this token belongs to.
        token_idx: Position of this token in the sequence.
        score: MaxSim score for this patch.
        align_target: Aligned key-patch index (-1 if none).
        side: ``"query"`` or ``"doc"``.
        pair_idx: Pair identifier string.
        min_s, max_s: Score range used to compute opacity.
        cut_prob: Optional raw boundary probability from the predictor.
            When provided, it is appended to the tooltip title.

    Returns:
        HTML ``<span>`` element.
    """
    rgb = PATCH_COLORS[patch_idx % len(PATCH_COLORS)]
    opacity = 0.2 + 0.8 * (score - min_s) / (max_s - min_s + 1e-9)
    bg = f"rgba({rgb},{opacity:.3f})"
    tok_clean = token.replace("Ġ", " ")
    attrs = (
        f"data-pair='{pair_idx}' data-side='{side}' "
        f"data-patchidx='{patch_idx}' data-tokenidx='{token_idx}' "
        f"data-align='{align_target}' data-bg='{bg}'"
    )
    cp_suffix = f" | cut_prob={cut_prob:.2f}" if cut_prob is not None else ""
    return (
        f"<span {attrs} style='background-color:{bg};padding:1px 3px;"
        f"border-radius:2px;cursor:crosshair;display:inline-block;"
        f"white-space:pre-wrap;transition:background-color 0.1s;border-left:"
        f"2px solid rgba({rgb},0.8);margin-left:1px;' "
        f"title='Patch {patch_idx} | Score: {score:.4f}{cp_suffix}'>"
        f"{tok_clean}</span>"
    )


JS_SCRIPT = """
<script>
document.querySelectorAll('span[data-side]').forEach(span => {
  span.addEventListener('mouseenter', () => {
    const pair  = span.dataset.pair;
    const side  = span.dataset.side;
    const patch = parseInt(span.dataset.patchidx);
    const align = parseInt(span.dataset.align);

    if (side === 'query' && align >= 0) {
      document.querySelectorAll(
        `span[data-pair='${pair}'][data-side='doc'][data-patchidx='${align}']`
      ).forEach(t => { t.style.backgroundColor='rgba(255,215,0,0.95)'; t.style.color='#000'; });
    }
    if (side === 'doc') {
      document.querySelectorAll(
        `span[data-pair='${pair}'][data-side='query'][data-align='${patch}']`
      ).forEach(q => { q.style.backgroundColor='rgba(255,215,0,0.95)'; q.style.color='#000'; });
    }
    span.style.outline = '2px solid #222';
  });
  span.addEventListener('mouseleave', () => {
    const pair = span.dataset.pair;
    document.querySelectorAll(`span[data-pair='${pair}']`).forEach(s => {
      s.style.backgroundColor = s.dataset.bg;
      s.style.color = '';
      s.style.outline = '';
    });
  });
});
</script>
"""


def generate_pair_html(
    q_tokens: List[str],
    d_tokens: List[str],
    q_spans: List[List[int]],
    d_spans: List[List[int]],
    q_scores: torch.Tensor,             # (Pq,) per-patch score
    d_scores: List[float],              # per-token max received score
    align_1d: torch.Tensor,             # (Pq,) aligned doc-patch index per query patch
    q_mask: torch.Tensor,
    d_mask: torch.Tensor,
    pos_score: float,
    neg_score: float,
    pair_idx: str,
    q_cut_probs: Optional[torch.Tensor] = None,   # (S,) optional, for tooltip + heatmap
    diagnostic_heatmap: bool = False,
) -> str:
    """Render a single (query, positive-doc) pair as an HTML block.

    Args:
        q_tokens: Decoded query token strings.
        d_tokens: Decoded document token strings.
        q_spans: Token-index lists per query patch.
        d_spans: Token-index lists per doc patch.
        q_scores: ``(Pq,)`` per-patch MaxSim scores.
        d_scores: Per-token maximum received score.
        align_1d: ``(Pq,)`` aligned doc-patch index per query patch.
        q_mask, d_mask: Attention masks (1-D, already indexed ``[0]``).
        pos_score, neg_score: Scalar retrieval scores.
        pair_idx: Identifier string used for CSS data attributes.
        q_cut_probs: When provided, each token's cut_prob is added to its
            tooltip title.  Required when ``diagnostic_heatmap=True``.
        diagnostic_heatmap: When ``True``, prepend a per-token cut_prob
            heatmap block above the patch visualisation.

    Returns:
        HTML string for this pair.
    """
    all_scores = q_scores[q_scores > -100].tolist() + [s for s in d_scores if s > 0]
    if not all_scores:
        return ""
    min_s, max_s = min(all_scores), max(all_scores)

    correct = "✓" if pos_score > neg_score else "✗"
    color = "#2a9d2a" if pos_score > neg_score else "#d62728"
    badge = (
        f"<span style='font-size:13px;color:{color};font-weight:bold;margin-left:12px'>"
        f"{correct} pos={pos_score:.3f} neg={neg_score:.3f}</span>"
    )

    # Build per-token cut_prob lookup (None if not available)
    cp_lookup: Optional[List[float]] = (
        q_cut_probs.cpu().tolist() if q_cut_probs is not None else None
    )

    # Query spans HTML
    q_html_parts = []
    for pi, tok_indices in enumerate(q_spans):
        patch_score = float(q_scores[pi]) if pi < len(q_scores) else 0.0
        align_target = int(align_1d[pi]) if pi < len(align_1d) else -1
        for ti in tok_indices:
            if q_mask[ti] == 0:
                continue
            cp = cp_lookup[ti] if cp_lookup is not None and ti < len(cp_lookup) else None
            q_html_parts.append(_patch_span_html(
                q_tokens[ti], pi, ti, patch_score, align_target,
                "query", pair_idx, min_s, max_s, cut_prob=cp,
            ))

    # Doc spans HTML
    d_html_parts = []
    for pi, tok_indices in enumerate(d_spans):
        doc_patch_score = max((d_scores[ti] for ti in tok_indices
                               if ti < len(d_scores)), default=0.0)
        for ti in tok_indices:
            if ti >= len(d_tokens) or d_mask[ti] == 0:
                continue
            d_html_parts.append(_patch_span_html(
                d_tokens[ti], pi, ti, doc_patch_score, -1,
                "doc", pair_idx, min_s, max_s,
            ))

    # Optional diagnostic heatmap block (query side only)
    heatmap_html = ""
    if diagnostic_heatmap and q_cut_probs is not None:
        from deep_stylometry.experiments._patch_diagnostics import cut_prob_heatmap_html
        heatmap_html = cut_prob_heatmap_html(
            q_tokens, q_cut_probs, q_mask, pair_idx
        )

    return f"""
    <div style='margin-bottom:32px;padding:20px;border:1px solid #ccc;
                border-radius:8px;font-family:sans-serif;'>
      <h3 style='margin-top:0'>Pair {pair_idx} {badge}</h3>
      {heatmap_html}
      <p style='font-size:13px;color:#666;font-weight:bold;margin:0 0 8px'>
        Query (patch boundaries = left border, hover to see alignment)</p>
      <div style='line-height:2.4;font-size:15px;margin-bottom:20px'>{''.join(q_html_parts)}</div>
      <p style='font-size:13px;color:#666;font-weight:bold;margin:0 0 8px'>Target Document</p>
      <div style='line-height:2.4;font-size:15px'>{''.join(d_html_parts)}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Statistics (unchanged from original)
# ---------------------------------------------------------------------------

def compute_patch_stats(all_patch_lengths: List[int]) -> dict:
    """Summarise a flat list of patch lengths.

    Args:
        all_patch_lengths: One entry per patch across all processed queries.

    Returns:
        Dict with mean, median, and top-10 most common lengths.
    """
    if not all_patch_lengths:
        return {}
    cnt = Counter(all_patch_lengths)
    total = len(all_patch_lengths)
    return {
        "mean": sum(all_patch_lengths) / total,
        "median": sorted(all_patch_lengths)[total // 2],
        "most_common": cnt.most_common(10),
    }


# ---------------------------------------------------------------------------
# PAN19 loader
# ---------------------------------------------------------------------------

def _load_pan19_triplets(
    cfg: Any,
    n_samples: int,
    seed: int = 42,
    language: str = "en",
    pan19_zip: Optional[str] = None,
) -> Any:  # returns datasets.Dataset with columns {query, positive, negative}
    """Sample (query, positive, negative) triplets from PAN19 deterministically.

    Uses ``PAN19Datamodule``'s static parsing and triplet-conversion methods
    directly — no datamodule instantiation, no tokenisation.  Tokenisation
    happens downstream in the main loop, identically to the HALvest path.

    Args:
        cfg: Loaded ``BaseConfig`` — accepted for API symmetry but currently
             unused (tokenisation settings are read in the caller).
        n_samples: Maximum number of triplets to return.
        seed: Random seed for negative-candidate selection inside
              ``PAN19Datamodule._convert_to_triplets``.
        language: PAN19 language filter (``"en"`` is the only supported value).
        pan19_zip: Path to the PAN19 ZIP archive.  Falls back to the
                   ``PAN19_ZIP`` environment variable (same convention as
                   ``train_utils.setup_datamodule``).

    Returns:
        A ``datasets.Dataset`` with columns ``{"query", "positive", "negative"}``,
        at most ``n_samples`` rows, with any empty-text triplets removed.

    Raises:
        ValueError: If neither ``pan19_zip`` nor ``PAN19_ZIP`` is set, or if
                    parsing produces zero triplets.
    """
    import datasets as _hf
    from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule

    zip_path = pan19_zip or os.environ.get("PAN19_ZIP")
    if not zip_path:
        raise ValueError(
            "PAN19 ZIP path required.  Pass --pan19_zip or set the PAN19_ZIP "
            "environment variable."
        )

    logger.info("Parsing PAN19 problems from %s (language=%s) …", zip_path, language)
    problems = PAN19Datamodule._parse_problems_from_zip(zip_path, language=language)

    # DECISION: use split=None (all problems) so we are not restricted to the
    # 20 % test split when n_samples << total.  We control the sample count
    # ourselves via .select(range(n_samples)) below.  This matches the spirit
    # of the HALvest path which reads from the training split.
    triplet_ds = PAN19Datamodule._convert_to_triplets(
        problems, seed=seed, split=None
    )

    n_available = len(triplet_ds)
    if n_available == 0:
        raise ValueError(
            "PAN19 parsing produced 0 triplets.  "
            "Check the ZIP path and language filter."
        )

    if n_available < n_samples:
        logger.warning(
            "PAN19: only %d triplets available; requested %d.  Using all.",
            n_available, n_samples,
        )
        n_samples = n_available

    triplet_ds = triplet_ds.select(range(n_samples))

    # Validate: drop any triplet with an empty text field.
    valid_indices = [
        i for i, row in enumerate(triplet_ds)
        if row["query"] and row["positive"] and row["negative"]
    ]
    n_dropped = len(triplet_ds) - len(valid_indices)
    if n_dropped > 0:
        logger.warning(
            "PAN19: dropping %d triplets with empty text fields.", n_dropped
        )
        triplet_ds = triplet_ds.select(valid_indices)

    logger.info(
        "PAN19: loaded %d triplets (seed=%d, language=%s).",
        len(triplet_ds), seed, language,
    )
    return triplet_ds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the patch-interaction analysis."""
    import datasets as hf_datasets
    from transformers import AutoTokenizer

    from deep_stylometry.modules import DeepStylometry
    from deep_stylometry.modules.patch_interaction import PatchInteraction
    from deep_stylometry.utils.configs import BaseConfig

    parser = argparse.ArgumentParser(
        description=(
            "Visualise learned-PLI patch interactions on HALvest-Contrastive "
            "or PAN19."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config_path", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint_path", required=True, help="Path to .ckpt checkpoint."
    )
    parser.add_argument(
        "--dataset",
        choices=["halvest", "pan19"],
        default="halvest",
        help=(
            "Dataset to analyse.  'halvest' uses HALvest-Contrastive (see "
            "--subset); 'pan19' uses PAN 2019 CDAA (--subset is ignored)."
        ),
    )
    parser.add_argument(
        "--subset",
        default="base-2",
        help=(
            "HALvest-Contrastive subset to load (e.g. 'base-2', 'base-10').  "
            "Ignored when --dataset pan19."
        ),
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of triplets to process.",
    )
    parser.add_argument(
        "--n_viz",
        type=int,
        default=None,
        help=(
            "Number of pairs to render in the HTML visualisation.  "
            "Defaults to 50 for halvest and 20 for pan19 (PAN19 spans are "
            "~512 tokens post-truncation, making each rendered pair large)."
        ),
    )
    parser.add_argument(
        "--output_html",
        default=None,
        help=(
            "Output HTML file path.  Defaults to 'interactions_patch.html' for "
            "halvest and 'interactions_patch_pan19.html' for pan19.  If passed "
            "explicitly, the given path is used verbatim regardless of dataset."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for PAN19 triplet sampling (negative-candidate "
            "selection).  Has no effect on the halvest path."
        ),
    )
    parser.add_argument(
        "--pan19_language",
        default="en",
        help="PAN19 language filter.  Only 'en' is currently supported.",
    )
    parser.add_argument(
        "--pan19_zip",
        default=None,
        help=(
            "Path to the PAN19 ZIP archive.  Overrides the PAN19_ZIP "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        default=False,
        help=(
            "Enable diagnostic mode.  Collects and reports per-token cut-probability "
            "statistics, patch-length distributions under all four inference rules, "
            "predictor sanity checks, and forward/direct cross-check.  Also adds a "
            "per-token cut_prob heatmap to the first %d HTML pairs.  "
            "When off, behaviour is identical to the normal fast path."
        ) % _N_DIAG_VIZ,
    )
    parser.add_argument(
        "--inference_mode",
        choices=["threshold_0.5", "threshold_0.3", "threshold_0.7", "gumbel"],
        default="threshold_0.5",
        help=(
            "Rule for deriving cut decisions from raw cut probabilities.  "
            "'threshold_0.5' is the default and the only mode used during "
            "training-time evaluation (TestEvalCallback); the other three "
            "exist only for diagnostic comparison and do not represent a "
            "recommendation about production inference behaviour.  "
            "'gumbel' re-runs the predictor with training=True (stochastic)."
        ),
    )
    args = parser.parse_args()

    # Resolve dataset-dependent defaults
    dataset_name: str = args.dataset
    n_viz: int = args.n_viz if args.n_viz is not None else (
        50 if dataset_name == "halvest" else 20
    )
    output_html: str = args.output_html or (
        "interactions_patch.html"
        if dataset_name == "halvest"
        else "interactions_patch_pan19.html"
    )
    output_dir = Path(output_html).parent

    logger.info(
        "patch_interactions: dataset=%s  n_samples=%d  seed=%d  "
        "n_viz=%d  output=%s  diagnostic=%s  inference_mode=%s",
        dataset_name, args.n_samples, args.seed, n_viz, output_html,
        args.diagnostic, args.inference_mode,
    )

    cfg = BaseConfig.from_yaml(args.config_path)
    model = DeepStylometry.load_from_checkpoint(args.checkpoint_path, cfg=cfg)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)

    # Use the trained PatchInteraction from the loaded model, not a fresh
    # instance.  A fresh PatchInteraction has a randomly-initialised
    # PatchBoundaryPredictor FFN, so learned patching silently degenerates
    # to nonsense boundaries.  See test_eval_callback.py "Fix 1" for the
    # canonical pattern.
    trained_pool = getattr(model.contrastive_loss, "pool", None)
    if isinstance(trained_pool, PatchInteraction):
        pli = trained_pool
        pli.eval()
        logger.info("Using trained PatchInteraction from model.contrastive_loss.pool.")
    else:
        logger.warning(
            "model.contrastive_loss.pool is not a PatchInteraction "
            "(got %s).  Falling back to a fresh PatchInteraction(cfg) — "
            "learned-PLI results will be meaningless.",
            type(trained_pool).__name__,
        )
        pli = PatchInteraction(cfg).to(device).eval()

    # --- Diagnostic: predictor sanity checks (before main loop) ---
    is_learned = (pli.predictor is not None)
    if args.diagnostic:
        if is_learned:
            from deep_stylometry.experiments._patch_diagnostics import (
                run_predictor_sanity_checks,
            )
            run_predictor_sanity_checks(pli, model, cfg)
        else:
            logger.warning(
                "[DIAGNOSTIC] pli.predictor is None — this is not a learned-PLI "
                "checkpoint.  Cut-probability diagnostics will be skipped."
            )
        # REVIEW: model.eval() is called before model.to(device) above.  This is
        # harmless (eval() only sets training-mode flags, no device dependency),
        # but the conventional ordering is to(device) first, then eval().
        logger.info(
            "[DIAGNOSTIC] attention_mask dtype check: tokenizer returns "
            "attention_mask as torch.LongTensor; predictor uses mask > 0 "
            "comparisons which are valid for both Long and Bool."
        )

    # --- Load dataset ---
    if dataset_name == "halvest":
        ds = hf_datasets.load_dataset(
            "almanach/halvest-contrastive", name=args.subset, split="train"
        ).select(range(min(args.n_samples, 999999)))
        dataset_label = f"HALvest-Contrastive ({args.subset})"
    else:
        ds = _load_pan19_triplets(
            cfg=cfg,
            n_samples=args.n_samples,
            seed=args.seed,
            language=args.pan19_language,
            pan19_zip=args.pan19_zip,
        )
        dataset_label = f"PAN19 ({args.pan19_language}, seed={args.seed})"

    # --- Diagnostic accumulators ---
    # These are only populated when --diagnostic is set and pli.predictor is not None.
    all_cut_probs_flat: List[float] = []
    # (cut_probs (1,S), mask (1,S)) pairs on CPU for cross-mode patch-length computation
    cut_probs_records: List[Tuple[torch.Tensor, torch.Tensor]] = []
    # Gumbel patch_ids per query, on CPU
    gumbel_patch_ids_records: List[torch.Tensor] = []

    # --- Shared tokenisation / patching / HTML loop ---
    all_patch_lengths: List[int] = []
    html_pairs: List[str] = []
    n_correct = 0
    cross_check_done = False

    def _tok(text: str) -> dict:
        return tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.data.max_length,
        ).to(device)

    for idx, row in enumerate(ds):
        q_tok = _tok(row["query"])
        pos_tok = _tok(row["positive"])
        neg_tok = _tok(row["negative"])

        with torch.no_grad():
            q_embs = model(q_tok["input_ids"], q_tok["attention_mask"])
            pos_embs = model(pos_tok["input_ids"], pos_tok["attention_mask"])
            neg_embs = model(neg_tok["input_ids"], neg_tok["attention_mask"])

        q_mask = q_tok["attention_mask"]
        pos_mask = pos_tok["attention_mask"]
        neg_mask = neg_tok["attention_mask"]

        # --- Diagnostic 4: forward cross-check (first triplet only) ---
        if args.diagnostic and not cross_check_done and is_learned:
            from deep_stylometry.experiments._patch_diagnostics import (
                run_forward_cross_check,
            )
            with torch.no_grad():
                run_forward_cross_check(
                    pli=pli,
                    q_embs=q_embs,
                    q_mask=q_mask,
                    q_input_ids=q_tok["input_ids"],
                    k_embs=pos_embs,
                    k_mask=pos_mask,
                    k_input_ids=pos_tok["input_ids"],
                )
            cross_check_done = True

        # --- Compute query patch IDs ---
        with torch.no_grad():
            if is_learned and (args.diagnostic or args.inference_mode != "threshold_0.5"):
                from deep_stylometry.experiments._patch_diagnostics import (
                    compute_patches_with_diagnostics,
                    _patch_ids_from_cut_probs,
                    INFERENCE_MODES,
                )
                q_ids, q_cut_probs_batch = compute_patches_with_diagnostics(
                    pli, q_embs, q_mask, q_tok["input_ids"], args.inference_mode
                )
            else:
                q_ids, _ = pli._compute_patches(
                    q_embs, q_mask, q_tok["input_ids"], step=None, training=False
                )
                q_cut_probs_batch = None

        # --- Diagnostic: accumulate cut_probs and cross-mode lengths ---
        if args.diagnostic and is_learned and q_cut_probs_batch is not None:
            # Valid (non-padding) positions only for population stats
            valid_mask = q_mask[0] > 0  # (S,)
            all_cut_probs_flat.extend(
                q_cut_probs_batch[0][valid_mask].detach().cpu().tolist()
            )
            cut_probs_records.append((
                q_cut_probs_batch.detach().cpu(),
                q_mask.cpu(),
            ))

            # Collect gumbel patch_ids (separate predictor call with training=True)
            with torch.no_grad():
                g_ids, _, _ = pli.predictor(q_embs, q_mask, step=None, training=True)
            gumbel_patch_ids_records.append(g_ids.detach().cpu())

        with torch.no_grad():
            pos_ids, _ = pli._compute_patches(
                pos_embs, pos_mask, pos_tok["input_ids"], step=None, training=False
            )

        q_max_p = max(1, int(q_ids.clamp(min=0).max().item()) + 1)
        pos_max_p = max(1, int(pos_ids.clamp(min=0).max().item()) + 1)

        with torch.no_grad():
            q_patch_embs, q_patch_mask = pli._compress_patches(
                q_embs, q_ids, q_mask, q_max_p
            )
            pos_patch_embs, pos_patch_mask = pli._compress_patches(
                pos_embs, pos_ids, pos_mask, pos_max_p
            )

        scores_1d, align_1d, pos_score = compute_patch_alignments(
            q_patch_embs, pos_patch_embs, q_patch_mask, pos_patch_mask
        )

        with torch.no_grad():
            neg_ids, _ = pli._compute_patches(
                neg_embs, neg_mask, neg_tok["input_ids"], step=None, training=False
            )
            neg_max_p = max(1, int(neg_ids.clamp(min=0).max().item()) + 1)
            neg_patch_embs, neg_patch_mask = pli._compress_patches(
                neg_embs, neg_ids, neg_mask, neg_max_p
            )
        _, _, neg_score = compute_patch_alignments(
            q_patch_embs, neg_patch_embs, q_patch_mask, neg_patch_mask
        )

        if pos_score > neg_score:
            n_correct += 1

        for pid in range(q_max_p):
            patch_len = int(
                ((q_ids[0] == pid) & (q_mask[0] > 0)).sum().item()
            )
            if patch_len > 0:
                all_patch_lengths.append(patch_len)

        if idx < n_viz:
            q_tokens = tokenizer.convert_ids_to_tokens(
                q_tok["input_ids"][0].tolist()
            )
            d_tokens = tokenizer.convert_ids_to_tokens(
                pos_tok["input_ids"][0].tolist()
            )
            q_spans = get_patch_spans(q_tokens, q_ids[0], q_mask[0])
            d_spans = get_patch_spans(d_tokens, pos_ids[0], pos_mask[0])

            d_scores_list = [0.0] * len(d_tokens)
            for qi in range(min(len(scores_1d), q_max_p)):
                if q_patch_mask[0, qi] == 0:
                    continue
                doc_pi = int(align_1d[qi].item())
                score_val = float(scores_1d[qi].item())
                if doc_pi < len(d_spans):
                    for ti in d_spans[doc_pi]:
                        if ti < len(d_scores_list):
                            d_scores_list[ti] = max(d_scores_list[ti], score_val)

            # Pass cut_probs for tooltip enrichment and heatmap (diagnostic only)
            q_cp_for_html = (
                q_cut_probs_batch[0].detach().cpu()
                if (args.diagnostic and q_cut_probs_batch is not None)
                else None
            )
            pair_html = generate_pair_html(
                q_tokens=q_tokens,
                d_tokens=d_tokens,
                q_spans=q_spans,
                d_spans=d_spans,
                q_scores=scores_1d,
                d_scores=d_scores_list,
                align_1d=align_1d,
                q_mask=q_mask[0],
                d_mask=pos_mask[0],
                pos_score=float(pos_score),
                neg_score=float(neg_score),
                pair_idx=f"pair_{idx}",
                q_cut_probs=q_cp_for_html,
                diagnostic_heatmap=(args.diagnostic and idx < _N_DIAG_VIZ),
            )
            html_pairs.append(pair_html)

    n_total = len(ds)
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    stats = compute_patch_stats(all_patch_lengths)
    logger.info(
        "Triplet accuracy: %d/%d = %.4f", n_correct, n_total, accuracy
    )
    logger.info("Patch stats: %s", stats)

    # --- Diagnostic summaries (printed to stdout, saved to JSON) ---
    if args.diagnostic and is_learned and all_cut_probs_flat:
        from deep_stylometry.experiments._patch_diagnostics import (
            print_and_save_cut_prob_stats,
            compute_and_save_patch_lengths_by_mode,
        )
        print_and_save_cut_prob_stats(
            all_cut_probs_flat,
            output_path=output_dir / "cut_probs_stats.json",
        )
        compute_and_save_patch_lengths_by_mode(
            cut_probs_records=cut_probs_records,
            gumbel_patch_ids_records=gumbel_patch_ids_records,
            output_path=output_dir / "patch_length_by_inference_mode.json",
        )
    elif args.diagnostic and not is_learned:
        logger.info(
            "[DIAGNOSTIC] Skipping cut_prob and patch-length stats — "
            "not a learned-PLI checkpoint."
        )

    # --- Print diagnostic block 3 (predictor summary) to stdout ---
    if args.diagnostic and is_learned:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 3 — Predictor Parameter Summary")
        print("=" * 70)
        print(f"  ffn[0].weight.dtype : {pli.predictor.ffn[0].weight.dtype}")
        print(f"  ffn[2].weight.dtype : {pli.predictor.ffn[2].weight.dtype}")
        n_params = sum(p.numel() for p in pli.predictor.parameters())
        print(f"  total parameters    : {n_params:,}")
        print()

    # --- Build HTML ---
    stats_html = (
        f"<div style='background:#f9f9f9;padding:12px;border-radius:6px;"
        f"font-family:monospace;font-size:13px;margin-bottom:24px;'>"
        f"<b>Dataset:</b> {dataset_label}<br>"
        f"<b>n_samples:</b> {n_total} &nbsp; <b>seed:</b> {args.seed}<br>"
        f"<b>inference_mode:</b> {args.inference_mode}<br>"
        f"<b>Triplet accuracy:</b> {n_correct}/{n_total} = {accuracy:.4f}<br>"
        f"<b>Mean patch length:</b> {stats.get('mean', 0):.2f} tokens<br>"
        f"<b>Median patch length:</b> {stats.get('median', 0)} tokens<br>"
        f"<b>Most common lengths:</b> "
        + ", ".join(f"{l}tok×{c}" for l, c in stats.get("most_common", [])[:8])
        + "</div>"
    )

    html = (
        "<html><head><meta charset='utf-8'>"
        "<title>PLI Patch Interactions</title></head>"
        "<body style='max-width:1100px;margin:0 auto;padding:20px;'>"
        "<h1>Patch-Level Late Interaction Analysis</h1>"
        + stats_html
        + "".join(html_pairs)
        + JS_SCRIPT
        + "</body></html>"
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Saved visualisation to %s", output_html)


if __name__ == "__main__":
    main()
