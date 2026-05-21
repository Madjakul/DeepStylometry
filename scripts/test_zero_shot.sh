#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

CONFIG_PATH=${1:-$PROJECT_ROOT/configs/test_zero_shot_e5_halvest.yml}
LOGS_DIR=$PROJECT_ROOT/logs
export PAN19_ZIP=${PAN19_ZIP:-$DATA_ROOT/pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23.zip}

# E5 prefixes (asymmetric retrieval convention).
# Override or unset to run without prefixes for an ablation.
# Uses ${VAR-default} (single dash): a user-supplied empty string is preserved,
# so `ZERO_SHOT_QUERY_PREFIX="" ZERO_SHOT_PASSAGE_PREFIX="" sbatch ...` works.
export ZERO_SHOT_QUERY_PREFIX="${ZERO_SHOT_QUERY_PREFIX-query: }"
export ZERO_SHOT_PASSAGE_PREFIX="${ZERO_SHOT_PASSAGE_PREFIX-passage: }"

# Prefixed and non-prefixed tokenisations produce different token IDs;
# they must not share a cache. The path segment below encodes this.
if [[ -n "$ZERO_SHOT_QUERY_PREFIX" || -n "$ZERO_SHOT_PASSAGE_PREFIX" ]]; then
    PROCESSED_DS_DIR=$SCRATCH/Datasets/deep-stylometry/intfloat-multilingual-e5-base-prefixed/no-padding/
else
    PROCESSED_DS_DIR=$SCRATCH/Datasets/deep-stylometry/intfloat-multilingual-e5-base/no-padding/
fi

NUM_PROC=16

# --------------------------------------------------------------------------------------

mkdir -p "$LOGS_DIR" || true

python3 "$PROJECT_ROOT/test_zero_shot.py" \
    --config_path "$CONFIG_PATH" \
    --processed_ds_dir "$PROCESSED_DS_DIR" \
    --logs_dir "$LOGS_DIR" \
    --num_proc "$NUM_PROC"
