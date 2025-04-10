# deep_stylometry/experiments/eda.py

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer


def generate_vocab_matrix(model_name="mistralai/Mistral-7B-v0.1"):
    # Load data and initialize
    ds = load_dataset("almanach/HALvest-Contrastive", "base-8", split="valid")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Build domain vocabularies with proper separation
    domain_vocabs = defaultdict(set)

    # for split in ["train", "valid", "test"]:
    for example in ds:  # [split]:
        # Process query
        query_tokens = set(tokenizer(example["query_text"])["input_ids"])
        for domain in example["query_domains"]:
            domain_vocabs[domain].update(query_tokens)

        # Process key
        key_tokens = set(tokenizer(example["key_text"])["input_ids"])
        for domain in example["key_domains"]:
            domain_vocabs[domain].update(key_tokens)

    # Matrix calculation with Jaccard similarity
    domains = sorted(domain_vocabs.keys())
    matrix = np.zeros((len(domains), len(domains)))

    for i, dom1 in enumerate(domains):
        for j, dom2 in enumerate(domains):
            intersection = len(domain_vocabs[dom1] & domain_vocabs[dom2])
            union = len(domain_vocabs[dom1] | domain_vocabs[dom2])
            matrix[i][j] = intersection / union if union > 0 else 0

    # Visualization with improved labeling
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(
        matrix,
        annot=True,
        xticklabels=domains,
        yticklabels=domains,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Domain Vocabulary Similarity (Jaccard Index)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    return matrix, domains


similarity_matrix, domain_labels = generate_vocab_matrix(
    model_name="openai-community/gpt2-xl"
)
plt.savefig("domain_similarity.png", dpi=300)
plt.show()
# plt.close()
