# experiments/mechanistic

Mechanistic interpretability pipeline for the paper
*Where Does Authorship Signal Emerge in Encoder-Based Language Models?*

## Purpose

This directory contains the code to trace where authorship-discriminative
information forms across the layers and training steps of contrastive encoder
models (ModernBERT-base fine-tuned with InfoNCE). The study uses five
complementary methods: linear probing of LISA features, residual patching
(causal intervention), training dynamics tracking, and distractor analysis.

## Configuration

All runs are configured via `configs/mechanistic.yml`. Edit that file to set
model checkpoint paths, dataset subsets, output directories, and phase
hyperparameters.

## Usage

### Run all phases sequentially

```bash
python -m deep_stylometry.experiments.mechanistic.run_phase \
    --config configs/mechanistic.yml --phase all
```

Or via the SLURM wrapper:

```bash
bash scripts/mechanistic.sh configs/mechanistic.yml
```

### Run a specific phase

```bash
python -m deep_stylometry.experiments.mechanistic.run_phase \
    --config configs/mechanistic.yml --phase 2 \
    --models mean li pli_ngram2
```

## Phases

### Phase 0: Build probe set (`probe_set.py`)

Constructs a balanced, stratified triplet probe set from the HALvest-Contrastive
test split. Triplets are stratified by retrieval difficulty tier (rank=1,
rank=2-5, rank=6-20, rank>20) to ensure all difficulty levels are represented.
Output: `probe_set.jsonl`.

### Phase 1a: Extract LISA features (`lisa_features.py`)

Extracts Linguistic-Informed Stylometric Analysis (LISA) features for each
document in the probe set. Features include function-word frequencies,
punctuation densities, POS tag distributions, average sentence length, and
type-token ratio.

### Phase 1b: Linear probing (`linear_probes.py`, `activation_extractor.py`)

Extracts per-layer hidden states from the fine-tuned model and trains a
logistic regression probe at each layer to predict LISA feature values. The
layer at which probe accuracy plateaus corresponds to where stylistic
information is encoded.

### Phase 2: Residual patching (`residual_patching.py`, `nnsight_helpers.py`)

Performs causal residual-stream patching: corrupts the residual stream at a
specific layer with activations from a semantically similar but stylistically
different document and measures the drop in authorship discrimination (MaxSim
score) at the output. Produces layer-wise recovery curves that identify an
"inflection layer" where authorship signal becomes load-bearing.

### Phase 3: Training dynamics (`training_dynamics.py`)

Re-evaluates all probes and patching experiments across a sequence of
training checkpoints. Tracks how probe accuracy and recovery curves evolve
during training to determine when authorship signal crystallises.

### Phase 4: Distractor analysis (`distractor_analysis.py`)

Analyses the predictable failure modes identified by `retrieval_inspection.py`
through the lens of activation space. Checks whether distractors (rank 2-20
documents) are geometrically closer to queries in specific layers compared to
random negatives.

## Orchestrator

`run_phase.py` manages phase dependencies: later phases require outputs from
earlier phases and will error if the required inputs are not present. Use
`--phase all` to run the full pipeline in order.

## Output Layout

All outputs are written to the directory specified by `cfg.output_dir` in
`configs/mechanistic.yml`, organised by model name and phase:

```
<output_dir>/
  <model_name>/
    probe_set.jsonl
    lisa_features.jsonl
    activations/
      layer_<n>.h5
    probes/
      layer_<n>_<feature>.pkl
    patching/
      recovery_curves.json
    dynamics/
      step_<k>/
    distractor/
      analysis.json
```
