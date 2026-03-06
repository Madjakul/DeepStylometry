# deep_styometry/experiments/token_interactions.py

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import datasets
from collections import Counter

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

    # Get the max score for each query token and the index of the doc token it aligned with
    max_results = scores.max(dim=-1)
    return max_results.values[0].cpu().numpy(), max_results.indices[0].cpu().numpy()


def process_pair(q_text, pos_text, model, tokenizer, device):
    """Tokenizes, embeds, aligns, and returns mapped interaction data."""
    q_tok = tokenizer(q_text, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    pos_tok = tokenizer(
        pos_text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        q_embs = model(
            input_ids=q_tok["input_ids"], attention_mask=q_tok["attention_mask"]
        )
        pos_embs = model(
            input_ids=pos_tok["input_ids"], attention_mask=pos_tok["attention_mask"]
        )

        scores_1d, align_1d = get_inline_late_interaction(
            q_embs, pos_embs, q_tok["attention_mask"], pos_tok["attention_mask"]
        )

    q_tokens = tokenizer.convert_ids_to_tokens(q_tok["input_ids"][0])
    pos_tokens = tokenizer.convert_ids_to_tokens(pos_tok["input_ids"][0])

    # 1. Identify valid indices (exclude padding and special tokens like [CLS], [SEP])
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

    # 2. Build Query Data (The drivers of the score)
    query_data = []
    for qi in q_valid:
        tok = q_tokens[qi]
        clean_tok = tok.replace("Ġ", "").strip().lower()
        query_data.append(
            {"token_raw": tok, "token_clean": clean_tok, "score": float(scores_1d[qi])}
        )

    # 3. Build Document Data (The targets)
    doc_scores = [0.0] * len(pos_tokens)
    for q_idx, doc_idx in enumerate(align_1d):
        if q_tok["attention_mask"][0, q_idx] == 0:
            continue
        if scores_1d[q_idx] > doc_scores[doc_idx]:
            doc_scores[doc_idx] = float(scores_1d[q_idx])

    doc_data = []
    for di in doc_valid:
        tok = pos_tokens[di]
        clean_tok = tok.replace("Ġ", "").lower()
        doc_data.append(
            {"token_raw": tok, "token_clean": clean_tok, "score": float(doc_scores[di])}
        )

    # 4. Create remapped alignments for the Javascript cross-linking
    doc_valid_set = {orig_idx: new_idx for new_idx, orig_idx in enumerate(doc_valid)}

    align_remapped = [
        doc_valid_set.get(
            int(align_1d[qi]), -1
        )  # -1 if it aligned to a special/pad token
        for qi in q_valid
    ]

    return query_data, doc_data, align_remapped


def generate_html_heatmap_pair(q_data, doc_data, align_1d, pair_idx):
    """Generates the interactive HTML blocks with data-attributes for JS
    linking."""
    if not q_data or not doc_data:
        return ""

    # Find min/max scores to maintain the base heatmap colors
    all_scores = [item["score"] for item in q_data + doc_data]
    min_s, max_s = min(all_scores), max(all_scores)

    def _span(item, side, local_idx, align_target=None):
        score = item["score"]
        tok = item["token_raw"].replace("Ġ", " ")

        # Base Heatmap Colors
        color = "100,149,237" if side == "query" else "255,99,71"
        opacity = 0.15 + 0.85 * ((score - min_s) / (max_s - min_s + 1e-9))
        bg_color = f"rgba({color},{opacity})"

        attrs = (
            f"data-pair='{pair_idx}' data-side='{side}' data-idx='{local_idx}' "
            f"data-align='{align_target if align_target is not None else -1}' "
            f"data-bg='{bg_color}'"
        )

        return (
            f"<span {attrs} style='background-color:{bg_color};padding:1px 3px;"
            f"border-radius:2px;cursor:crosshair;display:inline-block;"
            f"white-space:pre-wrap;transition: background-color 0.1s;' "
            f"title='Score: {score:.4f}'>{tok}</span>"
        )

    q_spans = "".join(
        _span(item, "query", i, int(align_1d[i])) for i, item in enumerate(q_data)
    )
    doc_spans = "".join(_span(item, "doc", i) for i, item in enumerate(doc_data))

    return f"""
    <div style='margin-bottom:32px;padding:20px;border:1px solid #ccc;border-radius:8px;font-family:sans-serif;'>
      <h3 style='margin-top:0'>Pair {pair_idx+1}</h3>
      <p style='font-size:13px;color:#666;font-weight:bold;margin:0 0 8px'>Query Document (Drivers of Score)</p>
      <div style='line-height:2.4;font-size:15px;margin-bottom:20px'>{q_spans}</div>
      <p style='font-size:13px;color:#666;font-weight:bold;margin:0 0 8px'>Target Document</p>
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
      const target = document.querySelector(`span[data-pair='${pair}'][data-side='doc'][data-idx='${align}']`);
      if (target) {
          target.style.backgroundColor = 'rgba(255,215,0,0.95)'; // Bright Gold
          target.style.color = '#000';
      }
    }
    if (side === 'doc') {
      document.querySelectorAll(`span[data-pair='${pair}'][data-side='query'][data-align='${idx}']`).forEach(q => {
          q.style.backgroundColor = 'rgba(255,215,0,0.95)'; // Bright Gold
          q.style.color = '#000';
      });
    }
    span.style.outline = '2px solid #222';
  });

  span.addEventListener('mouseleave', () => {
    const pair = span.dataset.pair;
    document.querySelectorAll(`span[data-pair='${pair}']`).forEach(s => {
      s.style.backgroundColor = s.dataset.bg; // Restore base heatmap color
      s.style.color = '';
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
    parser.add_argument("--subset", type=str, default="base-2")
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

    # Filter first, then slice safely to avoid crash
    pre_2024_ds = ds.filter(
        lambda x: 0 < parse_year(x["query_year"]) < 2023
        and 0 < parse_year(x["pos_year"]) < 2023
    )
    pre_2024_ds = pre_2024_ds.select(range(min(100, len(pre_2024_ds))))

    post_2023_ds = ds.filter(
        lambda x: parse_year(x["query_year"]) >= 2023
        and parse_year(x["pos_year"]) >= 2023
    )
    post_2023_ds = post_2023_ds.select(range(min(100, len(post_2023_ds))))

    html_output = "<html><head><title>Stylometry Tokens</title></head><body style='max-width: 1000px; margin: 0 auto; padding: 20px;'>"
    html_output += "<h1>Late Interaction Stylometry Analysis</h1>"

    for period_name, ds_sample in [
        ("Pre-2024 (Human Era)", pre_2024_ds),
        ("Post-2023 (LLM Era)", post_2023_ds),
    ]:
        print(f"\n{'='*40}\nAnalyzing: {period_name}\n{'='*40}")
        html_output += f"<h2 style='border-bottom: 2px solid #eee; padding-bottom: 10px;'>{period_name}</h2>"

        all_content_tokens = []

        for idx, row in enumerate(ds_sample):
            q_data, doc_data, align_remapped = process_pair(
                row["query"], row["positive"], model, tokenizer, device
            )

            # Visualize a few pairs for the HTML output
            if idx < 5:
                html_output += generate_html_heatmap_pair(
                    q_data,
                    doc_data,
                    align_remapped,
                    pair_idx=f"{'pre' if 'Pre' in period_name else 'post'}_{idx}",
                )

            # Experiment 2: Frequency analysis based on Q_DATA (Drivers of the score)
            q_data_sorted = sorted(q_data, key=lambda x: x["score"], reverse=True)
            top_k_cutoff = max(1, int(len(q_data_sorted) * 0.15))

            for item in q_data_sorted[:top_k_cutoff]:
                all_content_tokens.append(item["token_clean"])

        # Print the Stylistic Shift Results
        top_words = Counter(all_content_tokens).most_common(30)
        print(f"Top 30 tokens driving similarity in {period_name}:")
        for word, count in top_words:
            print(f"  - {word:<15}: {count} occurrences")

    html_output += JS_SCRIPT + "</body></html>"
    with open("interactions_heatmap.html", "w", encoding="utf-8") as f:
        f.write(html_output)
    print("\nSaved interactions_heatmap.html!")
