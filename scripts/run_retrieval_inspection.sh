#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

# First three positional args: CONFIG_PATH, CHECKPOINT_PATH, SUBSET
# Args after '--' are forwarded verbatim to retrieval_inspection.py
CONFIG_PATH=${1:-$PROJECT_ROOT/configs/test_pli_ngram2.yml}
CHECKPOINT_PATH=${2:-$PROJECT_ROOT/tmp/answerdotai-modernbert-base__halvest__pooling-pli-ngram-n2__skip_list-true/last.ckpt}
SUBSET=${3:-base-4}
OUTPUT_DIR=$PROJECT_ROOT/analysis/retrieval_inspection/${SUBSET}
LOGS_DIR=$PROJECT_ROOT/logs

# Collect extra args: everything after the '--' separator
EXTRA_ARGS=()
shift 3 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        shift
        EXTRA_ARGS+=("$@")
        break
    fi
    shift
done

# --------------------------------------------------------------------------------------

# CACHE_DIR=$SCRATCH/cache
NUM_PROC=16

# **************************************************************************************

mkdir -p "$LOGS_DIR" || true
mkdir -p "$OUTPUT_DIR" || true

if [[ $SLURM_JOB_ID != "" ]]; then
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "SLURM_NNODES: $SLURM_NNODES"
    echo "SLURM_NTASKS: $SLURM_NTASKS"
    echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "CONFIG_PATH: $CONFIG_PATH"
    echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
    echo "SUBSET: $SUBSET"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
fi

cmd=(
    python3 -m deep_stylometry.experiments.retrieval_inspection
    --config_path "$CONFIG_PATH"
    --checkpoint_path "$CHECKPOINT_PATH"
    --subset "$SUBSET"
    --output_dir "$OUTPUT_DIR"
    --n_seeds 5
    --n_queries_per_seed 100
    --top_k 20
    --batch_size 32
    --precision bf16-mixed
)

if [[ -v CACHE_DIR ]]; then
    cmd+=(--cache_dir "$CACHE_DIR")
fi

cmd+=("${EXTRA_ARGS[@]}")
"${cmd[@]}"
