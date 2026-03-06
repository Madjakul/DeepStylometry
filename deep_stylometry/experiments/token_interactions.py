# deep_styometry/experiments/token_interactions.py

import argparse
from collections import Counter, defaultdict

import datasets
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from deep_stylometry.modules import DeepStylometry
from deep_stylometry.utils.configs import BaseConfig


def get_inline_late_interaction(q_embs, pos_embs, q_mask, pos_mask):
    """Recomputes Late Interaction inline exactly as it happens during
    training."""
    q_norm = F.normalize(q_embs, p=2, dim=-1)
    pos_norm = F.normalize(pos_embs, p=2, dim=-1)

    # Shape: (batch, q_len, pos_len)
    scores = torch.einsum("bsh, bth -> bst", q_norm, pos_norm)

    # Mask out padded document tokens so they are never selected
    mask_inv = (1.0 - pos_mask.float()).unsqueeze(1)  # (batch, 1, pos_len)
    scores = scores + (mask_inv * -10000.0)

    max_results = scores.max(dim=-1)
    return max_results.values[0].cpu().numpy(), max_results.indices[0].cpu().numpy()


def late_interaction_score(q_embs, doc_embs, q_mask, doc_mask):
    """Returns a single scalar late-interaction score for a query/document
    pair."""
    q_norm = F.normalize(q_embs, p=2, dim=-1)
    doc_norm = F.normalize(doc_embs, p=2, dim=-1)
    scores = torch.einsum("bsh, bth -> bst", q_norm, doc_norm)
    mask_inv = (1.0 - doc_mask.float()).unsqueeze(1)
    scores = scores + (mask_inv * -10000.0)
    # Sum of per-query-token MaxSim, masked by query attention
    maxsim = scores.max(dim=-1).values[0]  # (q_len,)
    q_mask_1d = q_mask[0].bool().cpu()
    return maxsim[q_mask_1d].sum().item()


def process_pair(q_text, pos_text, neg_text, model, tokenizer, device):
    """Tokenizes and embeds query, positive, and negative.

    Returns:
      query_data      : list of dicts (token_raw, token_clean, score, orig_idx)
      doc_data        : list of dicts (token_raw, token_clean, score)
      align_remapped  : list of ints — doc_data index each query token aligned to
      pos_score       : float — full late-interaction score(query, positive)
      neg_score       : float — full late-interaction score(query, negative)
    """

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

    # Valid (non-padding, non-special) indices
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

    # Query data
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

    # Document data — score = max score received from any query token
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

    # Remap align_1d indices from raw token positions to doc_data positions
    doc_valid_set = {orig: new for new, orig in enumerate(doc_valid)}
    align_remapped = [doc_valid_set.get(int(align_1d[qi]), -1) for qi in q_valid]

    return query_data, doc_data, align_remapped, pos_score, neg_score


def generate_html_heatmap_pair(
    q_data, doc_data, align_1d, pair_idx, pos_score, neg_score
):
    """Generates interactive HTML blocks with data-attributes for JS cross-
    linking."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--subset", type=str, default="base-10")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument(
        "--n_viz",
        type=int,
        default=5,
        help="Number of pairs to render in the HTML per period.",
    )
    parser.add_argument("--top_k_tokens", type=int, default=30)
    parser.add_argument(
        "--top_pct",
        type=float,
        default=0.15,
        help="Top fraction of query tokens to use for frequency analysis.",
    )
    args = parser.parse_args()

    print(f"Loading config from {args.config_path}...")
    cfg = BaseConfig(mode="test").from_yaml(args.config_path)

    print(f"Loading model from {args.checkpoint_path}...")
    model = DeepStylometry.load_from_checkpoint(args.checkpoint_path, cfg=cfg)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)

    print(f"Loading {args.subset} from HuggingFace...")
    ds = datasets.load_dataset(
        "almanach/halvest-contrastive", name=args.subset, split="test"
    )

    def parse_year(y):
        try:
            return int(y)
        except (ValueError, TypeError):
            return -1

    pre_ds = ds.filter(
        lambda x: 0 < parse_year(x["query_year"]) <= 2023
        and 0 < parse_year(x["pos_year"]) <= 2023
    )
    pre_ds = pre_ds.select(range(min(args.n_samples, len(pre_ds))))

    post_ds = ds.filter(
        lambda x: parse_year(x["query_year"]) >= 2024
        and parse_year(x["pos_year"]) >= 2024
    )
    post_ds = post_ds.select(range(min(args.n_samples, len(post_ds))))

    html_output = (
        "<html><head><meta charset='utf-8'>"
        "<title>Stylometry Token Interactions</title></head>"
        "<body style='max-width:1000px;margin:0 auto;padding:20px;'>"
        "<h1>Late Interaction Stylometry Analysis</h1>"
    )

    for period_name, ds_sample in [
        ("Pre-2024 (Human Era)", pre_ds),
        ("Post-2023 (LLM Era)", post_ds),
    ]:
        print(
            f"\n{'='*40}\nAnalyzing: {period_name} ({len(ds_sample)} samples)\n{'='*40}"
        )
        html_output += (
            f"<h2 style='border-bottom:2px solid #eee;padding-bottom:10px;'>"
            f"{period_name}</h2>"
        )

        # token → list of MaxSim scores (for mean score weighting)
        token_scores: dict[str, list] = defaultdict(list)
        bigram_counter: Counter = Counter()
        n_correct = 0

        for idx, row in enumerate(ds_sample):
            q_data, doc_data, align_remapped, pos_score, neg_score = process_pair(
                row["query"],
                row["positive"],
                row["negative"],
                model,
                tokenizer,
                device,
            )

            if pos_score > neg_score:
                n_correct += 1

            if idx < args.n_viz:
                pair_idx = f"{'pre' if 'Pre' in period_name else 'post'}_{idx}"
                html_output += generate_html_heatmap_pair(
                    q_data,
                    doc_data,
                    align_remapped,
                    pair_idx,
                    pos_score,
                    neg_score,
                )

            # Token frequency + score accumulation (query side = drivers of score)
            q_sorted = sorted(q_data, key=lambda x: x["score"], reverse=True)
            top_k = max(1, int(len(q_sorted) * args.top_pct))
            top_tokens = q_sorted[:top_k]

            for item in top_tokens:
                token_scores[item["token_clean"]].append(item["score"])

            for a, b in zip(top_tokens[:-1], top_tokens[1:]):
                bigram_counter[f"{a['token_clean']} {b['token_clean']}"] += 1

        # Accuracy
        accuracy = n_correct / len(ds_sample) if ds_sample else 0.0
        print(f"Accuracy (pos > neg): {n_correct}/{len(ds_sample)} = {accuracy:.4f}")

        # Rank tokens by count × mean_score
        token_stats = {
            tok: {"count": len(sc), "mean_score": sum(sc) / len(sc)}
            for tok, sc in token_scores.items()
        }
        ranked = sorted(
            token_stats.items(),
            key=lambda x: x[1]["count"] * x[1]["mean_score"],
            reverse=True,
        )[: args.top_k_tokens]

        print(f"\nTop {args.top_k_tokens} tokens (count × mean_score):")
        for tok, stat in ranked:
            print(
                f"  {tok:<20} count={stat['count']:>4}  mean={stat['mean_score']:.4f}"
            )

        print(f"\nTop {args.top_k_tokens} bigrams:")
        for bigram, count in bigram_counter.most_common(args.top_k_tokens):
            print(f"  {bigram:<30}: {count}")

        # Embed accuracy + top tokens summary into HTML
        html_output += (
            f"<div style='background:#f9f9f9;padding:12px;border-radius:6px;"
            f"font-family:monospace;font-size:13px;margin-bottom:24px;'>"
            f"<b>Accuracy:</b> {n_correct}/{len(ds_sample)} = {accuracy:.4f}<br>"
            f"<b>Top tokens (count × mean score):</b> "
            + ", ".join(
                f"{tok} ({s['count']}×{s['mean_score']:.2f})" for tok, s in ranked[:15]
            )
            + "</div>"
        )

    html_output += JS_SCRIPT + "</body></html>"
    with open("interactions_heatmap.html", "w", encoding="utf-8") as f:
        f.write(html_output)
    print("\nSaved interactions_heatmap.html!")
