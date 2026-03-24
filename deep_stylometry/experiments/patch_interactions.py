# deep_stylometry/experiments/patch_interactions.py
"""Patch-level interaction visualiser.

Analogous to ``token_interactions.py`` but operates at the patch level.
Renders an interactive HTML with patch spans, MaxSim alignments, and
per-patch score contributions.  Also outputs summary statistics:
  - Average patch length distribution
  - Patch boundary positions relative to POS tags (via spaCy if available)
  - Most common patch lengths

Usage
-----
python -m deep_stylometry.experiments.patch_interactions \\
    --config_path configs/test.yml \\
    --checkpoint_path /path/to/checkpoint.ckpt \\
    --subset base-2 \\
    --n_samples 200 \\
    --n_viz 50 \\
    --output_html interactions_patch.html
"""

import argparse
import logging
from collections import Counter
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Core analysis helpers
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
) -> str:
    rgb = PATCH_COLORS[patch_idx % len(PATCH_COLORS)]
    opacity = 0.2 + 0.8 * (score - min_s) / (max_s - min_s + 1e-9)
    bg = f"rgba({rgb},{opacity:.3f})"
    tok_clean = token.replace("Ġ", " ")
    attrs = (
        f"data-pair='{pair_idx}' data-side='{side}' "
        f"data-patchidx='{patch_idx}' data-tokenidx='{token_idx}' "
        f"data-align='{align_target}' data-bg='{bg}'"
    )
    return (
        f"<span {attrs} style='background-color:{bg};padding:1px 3px;"
        f"border-radius:2px;cursor:crosshair;display:inline-block;"
        f"white-space:pre-wrap;transition:background-color 0.1s;border-left:"
        f"2px solid rgba({rgb},0.8);margin-left:1px;' "
        f"title='Patch {patch_idx} | Score: {score:.4f}'>{tok_clean}</span>"
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
    q_scores: torch.Tensor,   # (Pq,) per-patch score
    d_scores: List[float],    # per-token max received score
    align_1d: torch.Tensor,   # (Pq,) aligned doc-patch index per query patch
    q_mask: torch.Tensor,
    d_mask: torch.Tensor,
    pos_score: float,
    neg_score: float,
    pair_idx: str,
) -> str:
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

    # Query spans HTML
    q_html_parts = []
    for pi, tok_indices in enumerate(q_spans):
        patch_score = float(q_scores[pi]) if pi < len(q_scores) else 0.0
        align_target = int(align_1d[pi]) if pi < len(align_1d) else -1
        for ti in tok_indices:
            if q_mask[ti] == 0:
                continue
            q_html_parts.append(_patch_span_html(
                q_tokens[ti], pi, ti, patch_score, align_target,
                "query", pair_idx, min_s, max_s
            ))

    # Doc spans HTML
    d_html_parts = []
    for pi, tok_indices in enumerate(d_spans):
        # doc score = max received from any query patch
        doc_patch_score = max((d_scores[ti] for ti in tok_indices
                               if ti < len(d_scores)), default=0.0)
        for ti in tok_indices:
            if ti >= len(d_tokens) or d_mask[ti] == 0:
                continue
            d_html_parts.append(_patch_span_html(
                d_tokens[ti], pi, ti, doc_patch_score, -1,
                "doc", pair_idx, min_s, max_s
            ))

    return f"""
    <div style='margin-bottom:32px;padding:20px;border:1px solid #ccc;
                border-radius:8px;font-family:sans-serif;'>
      <h3 style='margin-top:0'>Pair {pair_idx} {badge}</h3>
      <p style='font-size:13px;color:#666;font-weight:bold;margin:0 0 8px'>
        Query (patch boundaries = left border, hover to see alignment)</p>
      <div style='line-height:2.4;font-size:15px;margin-bottom:20px'>{''.join(q_html_parts)}</div>
      <p style='font-size:13px;color:#666;font-weight:bold;margin:0 0 8px'>Target Document</p>
      <div style='line-height:2.4;font-size:15px'>{''.join(d_html_parts)}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_patch_stats(all_patch_lengths: List[int]) -> dict:
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
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import datasets
    from transformers import AutoTokenizer
    from deep_stylometry.modules import DeepStylometry
    from deep_stylometry.modules.patch_interaction import PatchInteraction
    from deep_stylometry.utils.configs import BaseConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--subset", default="base-2")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_viz", type=int, default=50)
    parser.add_argument("--output_html", default="interactions_patch.html")
    args = parser.parse_args()

    cfg = BaseConfig.from_yaml(args.config_path)
    model = DeepStylometry.load_from_checkpoint(args.checkpoint_path, cfg=cfg)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)
    pli = PatchInteraction(cfg).to(device).eval()

    ds = datasets.load_dataset(
        "almanach/halvest-contrastive", name=args.subset, split="train"
    ).select(range(min(args.n_samples, 999999)))

    all_patch_lengths: List[int] = []
    html_pairs = []
    n_correct = 0

    for idx, row in enumerate(ds):
        def _tok(text):
            return tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=512).to(device)

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

        # Compute patch IDs
        with torch.no_grad():
            q_ids, _ = pli._compute_patches(
                q_embs, q_mask, q_tok["input_ids"], step=None, training=False
            )
            pos_ids, _ = pli._compute_patches(
                pos_embs, pos_mask, pos_tok["input_ids"], step=None, training=False
            )

        # Compress patches
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

        # Compare against negative
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

        # Patch length statistics
        for pid in range(q_max_p):
            patch_len = int(((q_ids[0] == pid) & (q_mask[0] > 0)).sum().item())
            if patch_len > 0:
                all_patch_lengths.append(patch_len)

        if idx < args.n_viz:
            q_tokens = tokenizer.convert_ids_to_tokens(q_tok["input_ids"][0].tolist())
            d_tokens = tokenizer.convert_ids_to_tokens(pos_tok["input_ids"][0].tolist())
            q_spans = get_patch_spans(q_tokens, q_ids[0], q_mask[0])
            d_spans = get_patch_spans(d_tokens, pos_ids[0], pos_mask[0])

            # Compute doc-side scores
            d_scores = [0.0] * len(d_tokens)
            for qi in range(min(len(scores_1d), q_max_p)):
                if q_patch_mask[0, qi] == 0:
                    continue
                doc_pi = int(align_1d[qi].item())
                score_val = float(scores_1d[qi].item())
                if doc_pi < len(d_spans):
                    for ti in d_spans[doc_pi]:
                        if ti < len(d_scores):
                            d_scores[ti] = max(d_scores[ti], score_val)

            pair_html = generate_pair_html(
                q_tokens=q_tokens,
                d_tokens=d_tokens,
                q_spans=q_spans,
                d_spans=d_spans,
                q_scores=scores_1d,
                d_scores=d_scores,
                align_1d=align_1d,
                q_mask=q_mask[0],
                d_mask=pos_mask[0],
                pos_score=float(pos_score),
                neg_score=float(neg_score),
                pair_idx=f"pair_{idx}",
            )
            html_pairs.append(pair_html)

    accuracy = n_correct / len(ds) if len(ds) > 0 else 0.0
    stats = compute_patch_stats(all_patch_lengths)
    logging.info(f"Accuracy: {n_correct}/{len(ds)} = {accuracy:.4f}")
    logging.info(f"Patch stats: {stats}")

    stats_html = (
        f"<div style='background:#f9f9f9;padding:12px;border-radius:6px;"
        f"font-family:monospace;font-size:13px;margin-bottom:24px;'>"
        f"<b>Accuracy:</b> {n_correct}/{len(ds)} = {accuracy:.4f}<br>"
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

    with open(args.output_html, "w", encoding="utf-8") as f:
        f.write(html)
    logging.info(f"Saved visualisation to {args.output_html}")
