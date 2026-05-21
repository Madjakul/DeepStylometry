# deep_stylometry/experiments/token_interactions.py
"""Token-level late-interaction visualiser.

For HALvest-Contrastive the dataset is split into pre-2024 (human era) and
post-2023 (LLM era) periods; each period is analysed separately.

For PAN19 there is no publication-year metadata, so a single analysis pass is
run over the sampled triplets.

Usage
-----
# HALvest (default base-2 LI checkpoint)
python -m deep_stylometry.experiments.token_interactions \\
    --config_path configs/test.yml \\
    --checkpoint_path tmp/.../last.ckpt \\
    --subset base-2 --n_samples 400

# PAN19
python -m deep_stylometry.experiments.token_interactions \\
    --config_path configs/test.yml \\
    --checkpoint_path tmp/.../last.ckpt \\
    --dataset pan19 --n_samples 400 --seed 42 \\
    --output_html interactions_token_pan19.html
"""

import argparse
import logging
import os
from collections import Counter, defaultdict
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core analysis helpers (unchanged)
# ---------------------------------------------------------------------------

def get_inline_late_interaction(q_embs, pos_embs, q_mask, pos_mask):
    """Recomputes Late Interaction inline exactly as it happens during training."""
    q_norm = F.normalize(q_embs, p=2, dim=-1)
    pos_norm = F.normalize(pos_embs, p=2, dim=-1)

    scores = torch.einsum("bsh, bth -> bst", q_norm, pos_norm)
    mask_inv = (1.0 - pos_mask.float()).unsqueeze(1)
    scores = scores + (mask_inv * -10000.0)

    max_results = scores.max(dim=-1)
    return max_results.values[0].cpu().numpy(), max_results.indices[0].cpu().numpy()


def late_interaction_score(q_embs, doc_embs, q_mask, doc_mask):
    """Returns a single scalar late-interaction score for a query/document pair."""
    q_norm = F.normalize(q_embs, p=2, dim=-1)
    doc_norm = F.normalize(doc_embs, p=2, dim=-1)
    scores = torch.einsum("bsh, bth -> bst", q_norm, doc_norm)
    mask_inv = (1.0 - doc_mask.float()).unsqueeze(1)
    scores = scores + (mask_inv * -10000.0)
    maxsim = scores.max(dim=-1).values[0]
    q_mask_1d = q_mask[0].bool().cpu()
    return maxsim[q_mask_1d].sum().item()


def process_pair(q_text, pos_text, neg_text, model, tokenizer, device):

    def _tok(text):
        return tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
            device
        )

    q_tok = _tok(q_text)
    pos_tok = _tok(pos_text)
    neg_tok = _tok(neg_text)

    with torch.no_grad():
        q_embs = model(
            input_ids=q_tok["input_ids"], attention_mask=q_tok["attention_mask"]
        )
        pos_embs = model(
            input_ids=pos_tok["input_ids"], attention_mask=pos_tok["attention_mask"]
        )
        neg_embs = model(
            input_ids=neg_tok["input_ids"], attention_mask=neg_tok["attention_mask"]
        )

        scores_1d, align_1d = get_inline_late_interaction(
            q_embs, pos_embs, q_tok["attention_mask"], pos_tok["attention_mask"]
        )
        pos_score = late_interaction_score(
            q_embs, pos_embs, q_tok["attention_mask"], pos_tok["attention_mask"]
        )
        neg_score = late_interaction_score(
            q_embs, neg_embs, q_tok["attention_mask"], neg_tok["attention_mask"]
        )

    q_tokens = tokenizer.convert_ids_to_tokens(q_tok["input_ids"][0])
    pos_tokens = tokenizer.convert_ids_to_tokens(pos_tok["input_ids"][0])

    q_valid = [
        i
        for i, (tok, m) in enumerate(zip(q_tokens, q_tok["attention_mask"][0]))
        if m == 1 and tok not in tokenizer.all_special_tokens
    ]
    doc_valid = [
        i
        for i, (tok, m) in enumerate(zip(pos_tokens, pos_tok["attention_mask"][0]))
        if m == 1 and tok not in tokenizer.all_special_tokens
    ]

    query_data = []
    for qi in q_valid:
        tok = q_tokens[qi]
        query_data.append(
            {
                "token_raw": tok,
                "token_clean": tok.replace("Ġ", " ").strip().lower(),
                "score": float(scores_1d[qi]),
            }
        )

    doc_scores = [0.0] * len(pos_tokens)
    for q_idx, doc_idx in enumerate(align_1d):
        if q_tok["attention_mask"][0, q_idx] == 0:
            continue
        if scores_1d[q_idx] > doc_scores[doc_idx]:
            doc_scores[doc_idx] = float(scores_1d[q_idx])

    doc_data = []
    for di in doc_valid:
        tok = pos_tokens[di]
        doc_data.append(
            {
                "token_raw": tok,
                "token_clean": tok.replace("Ġ", " ").strip().lower(),
                "score": float(doc_scores[di]),
            }
        )

    doc_valid_set = {orig: new for new, orig in enumerate(doc_valid)}
    align_remapped = [doc_valid_set.get(int(align_1d[qi]), -1) for qi in q_valid]

    return query_data, doc_data, align_remapped, pos_score, neg_score


# ---------------------------------------------------------------------------
# HTML generation (unchanged)
# ---------------------------------------------------------------------------

def generate_html_heatmap_pair(
    q_data, doc_data, align_1d, pair_idx, pos_score, neg_score
):
    """Generates interactive HTML blocks with data-attributes for JS cross-linking."""
    if not q_data or not doc_data:
        return ""

    all_scores = [item["score"] for item in q_data + doc_data]
    min_s, max_s = min(all_scores), max(all_scores)

    correct = "✓" if pos_score > neg_score else "✗"
    color = "#2a9d2a" if pos_score > neg_score else "#d62728"
    accuracy_badge = (
        f"<span style='font-size:13px;color:{color};font-weight:bold;margin-left:12px'>"
        f"{correct} pos={pos_score:.3f} neg={neg_score:.3f}</span>"
    )

    def _span(item, side, local_idx, align_target=None):
        score = item["score"]
        tok = item["token_raw"].replace("Ġ", " ")
        rgb = "100,149,237" if side == "query" else "255,99,71"
        opacity = 0.15 + 0.85 * ((score - min_s) / (max_s - min_s + 1e-9))
        bg = f"rgba({rgb},{opacity:.3f})"
        attrs = (
            f"data-pair='{pair_idx}' data-side='{side}' data-idx='{local_idx}' "
            f"data-align='{align_target if align_target is not None else -1}' "
            f"data-bg='{bg}'"
        )
        return (
            f"<span {attrs} style='background-color:{bg};padding:1px 3px;"
            f"border-radius:2px;cursor:crosshair;display:inline-block;"
            f"white-space:pre-wrap;transition:background-color 0.1s;' "
            f"title='Score: {score:.4f}'>{tok}</span>"
        )

    q_spans = "".join(
        _span(item, "query", i, int(align_1d[i])) for i, item in enumerate(q_data)
    )
    doc_spans = "".join(_span(item, "doc", i) for i, item in enumerate(doc_data))

    return f"""
    <div style='margin-bottom:32px;padding:20px;border:1px solid #ccc;
                border-radius:8px;font-family:sans-serif;'>
      <h3 style='margin-top:0'>Pair {pair_idx} {accuracy_badge}</h3>
      <p style='font-size:13px;color:#666;font-weight:bold;margin:0 0 8px'>
        Query Document (Drivers of Score)</p>
      <div style='line-height:2.4;font-size:15px;margin-bottom:20px'>{q_spans}</div>
      <p style='font-size:13px;color:#666;font-weight:bold;margin:0 0 8px'>
        Target Document</p>
      <div style='line-height:2.4;font-size:15px'>{doc_spans}</div>
    </div>
    """


JS_SCRIPT = """
<script>
document.querySelectorAll('span[data-side]').forEach(span => {
  span.addEventListener('mouseenter', () => {
    const pair  = span.dataset.pair;
    const side  = span.dataset.side;
    const idx   = parseInt(span.dataset.idx);
    const align = parseInt(span.dataset.align);

    if (side === 'query' && align >= 0) {
      const target = document.querySelector(
        `span[data-pair='${pair}'][data-side='doc'][data-idx='${align}']`
      );
      if (target) {
        target.style.backgroundColor = 'rgba(255,215,0,0.95)';
        target.style.color = '#000';
      }
    }
    if (side === 'doc') {
      document.querySelectorAll(
        `span[data-pair='${pair}'][data-side='query'][data-align='${idx}']`
      ).forEach(q => {
        q.style.backgroundColor = 'rgba(255,215,0,0.95)';
        q.style.color = '#000';
      });
    }
    span.style.outline = '2px solid #222';
  });

  span.addEventListener('mouseleave', () => {
    const pair = span.dataset.pair;
    document.querySelectorAll(`span[data-pair='${pair}']`).forEach(s => {
      s.style.backgroundColor = s.dataset.bg;
      s.style.color   = '';
      s.style.outline = '';
    });
  });
});
</script>
"""


# ---------------------------------------------------------------------------
# PAN19 loader (mirrors patch_interactions._load_pan19_triplets)
# ---------------------------------------------------------------------------

def _load_pan19_triplets(
    cfg: Any,
    n_samples: int,
    seed: int = 42,
    language: str = "en",
    pan19_zip: Optional[str] = None,
) -> Any:  # returns datasets.Dataset with columns {query, positive, negative}
    """Sample (query, positive, negative) triplets from PAN19 deterministically.

    Uses ``PAN19Datamodule`` static methods directly — no datamodule
    instantiation, no tokenisation.

    Args:
        cfg: Accepted for API symmetry; not used inside the helper.
        n_samples: Maximum number of triplets to return.
        seed: Random seed for negative-candidate selection.
        language: PAN19 language filter (``"en"`` is the only supported value).
        pan19_zip: Path to the PAN19 ZIP archive. Falls back to the
                   ``PAN19_ZIP`` environment variable.

    Raises:
        ValueError: If no ZIP path is available or parsing yields 0 triplets.
    """
    from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule

    zip_path = pan19_zip or os.environ.get("PAN19_ZIP")
    if not zip_path:
        raise ValueError(
            "PAN19 ZIP path required. Pass --pan19_zip or set the PAN19_ZIP "
            "environment variable."
        )

    logger.info("Parsing PAN19 problems from %s (language=%s) …", zip_path, language)
    problems = PAN19Datamodule._parse_problems_from_zip(zip_path, language=language)

    # DECISION: use split=None (all problems) — same rationale as patch_interactions.
    triplet_ds = PAN19Datamodule._convert_to_triplets(problems, seed=seed, split=None)

    n_available = len(triplet_ds)
    if n_available == 0:
        raise ValueError(
            "PAN19 parsing produced 0 triplets. "
            "Check the ZIP path and language filter."
        )
    if n_available < n_samples:
        logger.warning(
            "PAN19: only %d triplets available; requested %d. Using all.",
            n_available, n_samples,
        )
        n_samples = n_available

    triplet_ds = triplet_ds.select(range(n_samples))

    valid_indices = [
        i for i, row in enumerate(triplet_ds)
        if row["query"] and row["positive"] and row["negative"]
    ]
    n_dropped = len(triplet_ds) - len(valid_indices)
    if n_dropped > 0:
        logger.warning("PAN19: dropping %d triplets with empty text fields.", n_dropped)
        triplet_ds = triplet_ds.select(valid_indices)

    logger.info(
        "PAN19: loaded %d triplets (seed=%d, language=%s).",
        len(triplet_ds), seed, language,
    )
    return triplet_ds


# ---------------------------------------------------------------------------
# Shared per-period analysis loop
# ---------------------------------------------------------------------------

def _run_period(
    period_name: str,
    ds_sample: Any,
    model: Any,
    tokenizer: Any,
    device: Any,
    n_viz: int,
    top_pct: float,
    top_k_tokens: int,
    period_tag: str,
) -> str:
    """Run the token-stats loop for one period/dataset slice; return HTML fragment."""
    print(f"\n{'='*40}\nAnalyzing: {period_name} ({len(ds_sample)} samples)\n{'='*40}")
    html = (
        f"<h2 style='border-bottom:2px solid #eee;padding-bottom:10px;'>"
        f"{period_name}</h2>"
    )

    token_scores: dict = defaultdict(list)
    bigram_counter: Counter = Counter()
    n_correct = 0

    for idx, row in enumerate(ds_sample):
        q_data, doc_data, align_remapped, pos_score, neg_score = process_pair(
            row["query"], row["positive"], row["negative"],
            model, tokenizer, device,
        )

        if pos_score > neg_score:
            n_correct += 1

        if idx < n_viz:
            pair_idx = f"{period_tag}_{idx}"
            html += generate_html_heatmap_pair(
                q_data, doc_data, align_remapped, pair_idx, pos_score, neg_score,
            )

        q_sorted = sorted(q_data, key=lambda x: x["score"], reverse=True)
        top_k = max(1, int(len(q_sorted) * top_pct))
        top_tokens = q_sorted[:top_k]

        for item in top_tokens:
            token_scores[item["token_clean"]].append(item["score"])
        for a, b in zip(top_tokens[:-1], top_tokens[1:]):
            bigram_counter[f"{a['token_clean']} {b['token_clean']}"] += 1

    accuracy = n_correct / len(ds_sample) if ds_sample else 0.0
    print(f"Triplet accuracy: {n_correct}/{len(ds_sample)} = {accuracy:.4f}")

    token_stats = {
        tok: {"count": len(sc), "mean_score": sum(sc) / len(sc)}
        for tok, sc in token_scores.items()
    }
    ranked = sorted(
        token_stats.items(),
        key=lambda x: x[1]["count"] * x[1]["mean_score"],
        reverse=True,
    )[:top_k_tokens]

    print(f"\nTop {top_k_tokens} tokens (count × mean_score):")
    for tok, stat in ranked:
        print(f"  {tok:<20} count={stat['count']:>4}  mean={stat['mean_score']:.4f}")

    print(f"\nTop {top_k_tokens} bigrams:")
    for bigram, count in bigram_counter.most_common(top_k_tokens):
        print(f"  {bigram:<30}: {count}")

    html += (
        f"<div style='background:#f9f9f9;padding:12px;border-radius:6px;"
        f"font-family:monospace;font-size:13px;margin-bottom:24px;'>"
        f"<b>Triplet accuracy:</b> {n_correct}/{len(ds_sample)} = {accuracy:.4f}<br>"
        f"<b>Top tokens (count × mean score):</b> "
        + ", ".join(
            f"{tok} ({s['count']}×{s['mean_score']:.2f})" for tok, s in ranked[:15]
        )
        + "</div>"
    )
    return html


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the token-interaction analysis."""
    import datasets as hf_datasets
    from transformers import AutoTokenizer

    from deep_stylometry.modules import DeepStylometry
    from deep_stylometry.utils.configs import BaseConfig

    parser = argparse.ArgumentParser(
        description=(
            "Visualise token-level late-interaction scores on "
            "HALvest-Contrastive or PAN19."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to YAML config.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to .ckpt checkpoint.")
    parser.add_argument(
        "--dataset",
        choices=["halvest", "pan19"],
        default="halvest",
        help=(
            "Dataset to analyse. 'halvest' uses HALvest-Contrastive split by "
            "publication year; 'pan19' uses PAN 2019 CDAA (--subset is ignored)."
        ),
    )
    parser.add_argument(
        "--subset", type=str, default="base-2",
        help="HALvest-Contrastive subset. Ignored when --dataset pan19.",
    )
    parser.add_argument("--n_samples", type=int, default=400,
                        help="Number of triplets to process.")
    parser.add_argument(
        "--n_viz", type=int, default=None,
        help=(
            "Number of pairs to render in HTML. "
            "Defaults to 50 for halvest and 20 for pan19."
        ),
    )
    parser.add_argument("--top_k_tokens", type=int, default=50,
                        help="Number of top tokens to report.")
    parser.add_argument(
        "--top_pct", type=float, default=0.15,
        help="Top fraction of query tokens to use for frequency analysis.",
    )
    parser.add_argument(
        "--output_html", default=None,
        help=(
            "Output HTML path. Defaults to 'interactions_heatmap.html' for "
            "halvest and 'interactions_heatmap_pan19.html' for pan19."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for PAN19 triplet sampling. No effect on halvest.",
    )
    parser.add_argument(
        "--pan19_language", default="en",
        help="PAN19 language filter. Only 'en' is currently supported.",
    )
    parser.add_argument(
        "--pan19_zip", default=None,
        help=(
            "Path to the PAN19 ZIP archive. "
            "Overrides the PAN19_ZIP environment variable."
        ),
    )
    args = parser.parse_args()

    dataset_name: str = args.dataset
    n_viz: int = args.n_viz if args.n_viz is not None else (
        50 if dataset_name == "halvest" else 20
    )
    output_html: str = args.output_html or (
        "interactions_heatmap.html"
        if dataset_name == "halvest"
        else "interactions_heatmap_pan19.html"
    )

    logger.info(
        "token_interactions: dataset=%s  n_samples=%d  seed=%d  "
        "n_viz=%d  output=%s",
        dataset_name, args.n_samples, args.seed, n_viz, output_html,
    )

    print(f"Loading config from {args.config_path}...")
    cfg = BaseConfig(mode="test").from_yaml(args.config_path)

    print(f"Loading model from {args.checkpoint_path}...")
    model = DeepStylometry.load_from_checkpoint(args.checkpoint_path, cfg=cfg)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)

    html_output = (
        "<html><head><meta charset='utf-8'>"
        "<title>Stylometry Token Interactions</title></head>"
        "<body style='max-width:1000px;margin:0 auto;padding:20px;'>"
        "<h1>Late Interaction Stylometry Analysis</h1>"
    )

    if dataset_name == "halvest":
        print(f"Loading {args.subset} from HuggingFace...")
        ds = hf_datasets.load_dataset(
            "almanach/halvest-contrastive", name=args.subset, split="train"
        )

        def parse_year(y):
            try:
                return int(y)
            except (ValueError, TypeError):
                return -1

        pre_ds = ds.filter(
            lambda x: 0 < parse_year(x["query_year"]) < 2023
            and 0 < parse_year(x["pos_year"]) < 2023
        )
        pre_ds = pre_ds.select(range(min(args.n_samples, len(pre_ds))))

        post_ds = ds.filter(
            lambda x: parse_year(x["query_year"]) >= 2023
            and parse_year(x["pos_year"]) >= 2023
        )
        post_ds = post_ds.select(range(min(args.n_samples, len(post_ds))))

        periods = [
            ("Pre-2024 (Human Era)", pre_ds, "pre"),
            ("Post-2023 (LLM Era)", post_ds, "post"),
        ]
    else:
        pan19_ds = _load_pan19_triplets(
            cfg=cfg,
            n_samples=args.n_samples,
            seed=args.seed,
            language=args.pan19_language,
            pan19_zip=args.pan19_zip,
        )
        dataset_label = f"PAN19 ({args.pan19_language}, seed={args.seed})"
        periods = [(dataset_label, pan19_ds, "pan19")]

    for period_name, ds_sample, period_tag in periods:
        html_output += _run_period(
            period_name=period_name,
            ds_sample=ds_sample,
            model=model,
            tokenizer=tokenizer,
            device=device,
            n_viz=n_viz,
            top_pct=args.top_pct,
            top_k_tokens=args.top_k_tokens,
            period_tag=period_tag,
        )

    html_output += JS_SCRIPT + "</body></html>"
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_output)
    print(f"\nSaved {output_html}")


if __name__ == "__main__":
    main()
