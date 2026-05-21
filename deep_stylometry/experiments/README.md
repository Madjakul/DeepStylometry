# experiments

Analysis and visualisation scripts for the HALvest-Contrastive paper.

Each script has a corresponding shell wrapper (and SLURM `.sbatch` file) in
`scripts/` and can be run locally or submitted to a SLURM cluster.

## Scripts

### `dataset_statistics.py`

Computes corpus-level statistics (token counts, sentence lengths, vocabulary
size, entropy, Jaccard overlap, word-length distributions) across all
HALvest-Contrastive subsets and PAN19. Outputs JSON summaries and PDF figures.

```bash
bash scripts/stats.sh configs/train.yml
```

### `per_field_eval.py`

Per-domain retrieval evaluation. Loads pre-computed score matrices produced by
`test.py` and breaks down Recall@k and nDCG@k by HAL scientific domain.

```bash
bash scripts/stats.sh configs/test.yml  # (wrapped in stats.sh)
```

### `retrieval_inspection.py`

Failure-mode analysis. Encodes the full test pool, globally ranks each query,
and computes statistics (author-set Jaccard, domain overlap, unigram overlap)
for distractors (rank 2-20) versus random baselines. Reports headline ratios
and rank distributions.

```bash
bash scripts/run_retrieval_inspection.sh configs/test.yml
```

### `retrieval_inspection_reaggregate.py`

Re-aggregates retrieval inspection results from an existing `pairs.jsonl`
file without re-running the full encoding pass. Useful when the aggregation
logic changes but the raw pair data is already computed.

### `patch_interactions.py`

Patch-level interaction analysis. Computes and visualises MaxSim scoring
patterns at different patch granularities, showing how much of the final score
comes from function-word patches versus content-word patches.

```bash
bash scripts/patch_interactions.sh configs/test_pli_wholeword.yml
```

### `token_interactions.py`

Token-level interaction analysis (full ColBERT-style). Visualises per-token
MaxSim contributions for selected query-document pairs.

```bash
bash scripts/token_interactions.sh configs/test.yml
```

### `semantic_decorrelation.py`

Measures how much semantic content remains in the model's embeddings relative
to a semantic baseline (E5). Computes cosine similarity distributions between
query and key embeddings split by same-domain vs. cross-domain pairs.

```bash
bash scripts/decorrelation.sh configs/decorrelation.yml
```

### `collaboration_statistics.py`

Author-set collaboration statistics for the co-authorship labelling design.
Computes the rate at which HALvest-Contrastive triplets involve shared authors
across the query, positive, and negative, and reports per-subset breakdowns.

```bash
bash scripts/collab_stats.sh
```
