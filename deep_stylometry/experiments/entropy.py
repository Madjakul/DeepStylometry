# deep_stylometry/analysis/entropy.py

import os
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from deep_stylometry.utils.data import DomainPerplexityDataModule


def compute_perplexity():
    # Disable tokenizer parallelism to prevent warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize data module
    data_module = DomainPerplexityDataModule(
        batch_size=16,
        num_proc=4,
        tokenizer_name="openai-community/gpt2",  # Match your model choice
        max_length=512,
        ds_name="almanach/HALvest-Contrastive",
    )
    data_module.prepare_data()
    data_module.setup("test")

    # Load model (using GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
    model.eval()

    # Dictionary to store metrics per domain
    domain_metrics = defaultdict(lambda: {"losses": [], "perplexities": []})

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            domains = batch["domains"]  # List of domain strings

            # Forward pass with labels for loss calculation
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)

            # Calculate per-token losses
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=model.config.pad_token_id
            )
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            losses = losses.view(shift_labels.size())

            # Mask out padding tokens
            valid_tokens = (shift_labels != model.config.pad_token_id).float()

            # Calculate per-example loss (averaged over valid tokens)
            per_example_loss = (losses * valid_tokens).sum(dim=1) / valid_tokens.sum(
                dim=1
            )

            # Aggregate by domain
            for loss_val, domain in zip(per_example_loss.cpu().numpy(), domains):
                domain_metrics[domain]["losses"].append(loss_val)
                domain_metrics[domain]["perplexities"].append(np.exp(loss_val))

    # Compute final statistics
    results = {}
    for domain, metrics in domain_metrics.items():
        avg_loss = np.mean(metrics["losses"])
        std_loss = np.std(metrics["losses"])
        results[domain] = {
            "perplexity": np.exp(avg_loss),
            "perplexity_std": np.std(metrics["perplexities"]),
            "loss": avg_loss,
            "loss_std": std_loss,
            "sample_count": len(metrics["losses"]),
        }

    # Print results sorted by sample count (most frequent first)
    print("\nDomain Perplexity Results:")
    print("=" * 50)
    for domain, stats in sorted(results.items(), key=lambda x: -x[1]["sample_count"]):
        print(f"\nDomain: {domain} (n={stats['sample_count']})")
        print(
            f"  Perplexity: {stats['perplexity']:.2f} ± {stats['perplexity_std']:.2f}"
        )
        print(f"  Loss: {stats['loss']:.4f} ± {stats['loss_std']:.4f}")
    print("=" * 50)
