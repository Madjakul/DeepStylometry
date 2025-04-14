# deep_stylometry/utils/data/preprocessing.py

import os

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, notebook_login  # Use notebook_login if in a notebook

# --- Configuration ---
original_ds_name = "AnnaWegmann/StyleEmbeddingData"
# IMPORTANT: Choose a unique name for your new dataset on the Hub
# It's usually good practice to prefix it with your username, e.g., "YourUsername/StyleEmbeddingPairwiseData"
new_ds_name = "YourUsername/StyleEmbeddingPairwiseData"  # <--- CHANGE THIS
num_proc = 4  # Adjust based on your available CPU cores

# --- Authentication (Required before pushing) ---
# Option 1: Run in terminal beforehand: huggingface-cli login
# Option 2: Run in a notebook:
# notebook_login()
# Option 3: Set HUGGING_FACE_HUB_TOKEN environment variable
print("Please ensure you are logged into Hugging Face Hub.")
print(
    "You can use 'huggingface-cli login' in your terminal or notebook_login() in a notebook."
)
# Add a check or prompt if needed, or rely on the push_to_hub error if not logged in.

# --- Load Original Dataset ---
print(f"Loading original dataset: {original_ds_name}...")
original_ds = load_dataset(original_ds_name)
print("Original dataset loaded.")
print(original_ds)


# --- Transformation Function ---
def transform_triplet_to_pairwise(batch):
    """Transforms a batch of triplet data into pairwise data.

    Each triplet (A, U1, U2, label) yields two pairs:
    - One positive pair (A, Positive Utterance) with label 1
    - One negative pair (A, Negative Utterance) with label 0
    """
    new_sentence1 = []
    new_sentence2 = []
    new_labels = []

    anchors = batch["Anchor (A)"]
    u1s = batch["Utterance 1 (U1)"]
    u2s = batch["Utterance 2 (U2)"]
    labels = batch["Same Author Label"]

    for anchor, u1, u2, label in zip(anchors, u1s, u2s, labels):
        anchor_str = str(anchor) if anchor is not None else ""
        u1_str = str(u1) if u1 is not None else ""
        u2_str = str(u2) if u2 is not None else ""

        if label == 1:
            # Positive Pair: A and U1
            new_sentence1.append(anchor_str)
            new_sentence2.append(u1_str)
            new_labels.append(1)
            # Negative Pair: A and U2
            new_sentence1.append(anchor_str)
            new_sentence2.append(u2_str)
            new_labels.append(0)
        elif label == 0:
            # Positive Pair: A and U2
            new_sentence1.append(anchor_str)
            new_sentence2.append(u2_str)
            new_labels.append(1)
            # Negative Pair: A and U1
            new_sentence1.append(anchor_str)
            new_sentence2.append(u1_str)
            new_labels.append(0)
        else:
            # Handle unexpected labels if necessary, here we'll skip them
            print(f"Warning: Unexpected label '{label}' found. Skipping entry.")
            continue

    return {
        "Sentence1": new_sentence1,
        "Sentence2": new_sentence2,
        "label": new_labels,
    }


# --- Apply Transformation to each split ---
print("Transforming dataset splits...")
transformed_ds = DatasetDict()
original_columns = original_ds["train"].column_names  # Get columns to remove

for split in original_ds.keys():
    print(f"Processing split: {split}...")
    transformed_ds[split] = original_ds[split].map(
        transform_triplet_to_pairwise,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,  # Remove old columns
        desc=f"Transforming {split} split",  # Progress bar description
    )

print("Transformation complete.")
print("Transformed dataset structure:")
print(transformed_ds)
print("\nExample entry (train split):")
print(transformed_ds["train"][0])
print(transformed_ds["train"][1])  # Show the pair generated from the same triplet

# --- Push to Hugging Face Hub ---
print(f"\nAttempting to push transformed dataset to Hub: {new_ds_name}")
try:
    transformed_ds.push_to_hub(new_ds_name)
    print(
        f"Successfully pushed dataset to: https://huggingface.co/datasets/{new_ds_name}"
    )
except Exception as e:
    print(f"\nError pushing dataset to Hub: {e}")
    print("Please ensure:")
    print(
        "1. You have replaced 'YourUsername/StyleEmbeddingPairwiseData' with your desired dataset name."
    )
    print(
        "2. You are logged into Hugging Face Hub (use 'huggingface-cli login' or notebook_login())."
    )
    print(
        "3. You have the 'huggingface_hub' library installed (`pip install huggingface_hub`)."
    )
    print(
        "4. The chosen dataset name doesn't already exist under your account (or set private=True/repo_id=... appropriately)."
    )
