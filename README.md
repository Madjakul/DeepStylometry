# DeepStylometry

Contrastive Authorship Attribution with Patch-Level Late Interaction.

[![arXiv](https://img.shields.io/badge/arXiv-2407.20595-b31b1b.svg)](https://arxiv.org/abs/2407.20595)
[![arXiv](https://img.shields.io/badge/arXiv-2605.19908-b31b1b.svg)](https://arxiv.org/abs/2605.19908)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Data-yellow)](https://huggingface.co/datasets/almanach/HALvest)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-ContrastiveData-yellow)](https://huggingface.co/datasets/almanach/halvest-contrastive)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Checkpoints-yellow)](https://huggingface.co/collections/Madjakul/deep-stylometry)

* See also: [HALvesting](https://github.com/Madjakul/HALvesting)
* See also: [HALvesting-Contrastive](https://github.com/Madjakul/HALvesting-Contrastive)



## Overview

This repository contains the code for two papers:

1. [**HALvest-Contrastive: Retrieval-Like Authorship Attribution with Patch-Level Late Interaction**](https://arxiv.org/abs/2407.20595).
   This paper introduces HALvest, a large multilingual scholarly corpus, and HALvest-Contrastive, an English contrastive authorship attribution benchmark. It proposes Patch-Level Late Interaction (PLI), a token-interaction architecture that groups subword tokens into patches before computing MaxSim.

2. [**Where Does Authorship Signal Emerge in Encoder-Based Language Models?**](https://arxiv.org/abs/2605.19908).
   A mechanistic interpretability study that traces where authorship-discriminative information forms across layers and training steps in contrastive encoder models.

## Installation

```bash
pip install -r requirements.txt
```

The installation sript [`install.sh`](./install.sh) was created for x86 architecture and installs flash-attention alongside everything else.

## Data Preparation

**HALvest-Contrastive** is loaded automatically from HuggingFace on first use. Pre-tokenized splits are cached to `$WORK_DIR/Datasets/deep-stylometry/answerdotai-modernbert-base/no-padding/` and reused on subsequent runs.

[**PAN19**](https://zenodo.org/records/3530313) requires downloading the evaluation zip from Zenodo (link above) and setting the `PAN19_ZIP` environment variable to point to the downloaded archive:

```bash
export PAN19_ZIP=/path/to/pan19-authorship-attribution-test-dataset-2019-11-19.zip
```

## Usage

### Training

```bash
bash scripts/train.sh configs/train_pli_wholeword.yml
```

### Testing (with checkpoint)

```bash
bash scripts/test.sh configs/test_pli_wholeword.yml --test_subset base-4
```

### Zero-shot evaluation (E5 baseline)

```bash
bash scripts/test_zero_shot.sh configs/test_zero_shot_e5_halvest.yml
```

## Configuration

Each experiment variant is controlled by a pair of YAML files:
`configs/train_<variant>.yml` for training and `configs/test_<variant>.yml` for evaluation.

Available pooling methods and variants:

| Variant | Description |
|---------|-------------|
| `mean` | Mean-pooling baseline (cosine similarity) |
| `li` | Full token-level ColBERT-style late interaction |
| `pli` + `wholeword` | PLI with whole-word patches |
| `pli` + `ngram-{2,3,4,5}` | PLI with n-gram patches of fixed size n |
| `pli` + `learned` | PLI with Gumbel-Softmax learned patch boundaries |

## Key Results

- Authorship attribution obeys the same empirical laws as information retrieval: InfoNCE loss, late-interaction architectures, and batch-size scaling all transfer.
- Full token-level interaction is not necessary. PLI with whole-word patches achieves competitive or superior cross-domain generalisation while being faster.
- Authorship signal emerges abruptly at a specific "inflection layer" in the encoder, rather than accumulating gradually.

## Analysis Scripts

The `experiments/` directory contains analysis and visualisation scripts for corpus statistics, retrieval failure analysis, patch interaction analysis, and dataset visualisations. See [`experiments/README.md`](experiments/README.md) for a description of each script.

The `experiments/mechanistic/` directory contains the mechanistic interpretability study pipeline (probing, residual patching, training dynamics, distractor analysis). See [`experiments/mechanistic/README.md`](experiments/mechanistic/README.md) for details.

## Citation

If you use this code or the HALvest-Contrastive dataset, please cite:

```bibtex
@misc{kulumba_halvest_2026,
      title={HALvest-Contrastive: Retrieval-Like Authorship Attribution with Patch-Level Late Interaction},
      author={Francis Kulumba and Wissam Antoun and Guillaume Vimont and Laurent Romary and Florian Cafiero},
      year={2026},
      eprint={2407.20595},
      archivePrefix={arXiv},
      primaryClass={cs.DL},
      url={https://arxiv.org/abs/2407.20595},
}
```

```bibtex
@misc{kulumba_does_2026,
      title={Where Does Authorship Signal Emerge in Encoder-Based Language Models?},
      author={Francis Kulumba and Guillaume Vimont and Laurent Romary and Florian Cafiero},
      year={2026},
      eprint={2605.19908},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.19908},
}
```
