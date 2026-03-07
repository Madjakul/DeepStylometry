#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

CONFIG_PATH=$PROJECT_ROOT/configs/test.yml
PROCESSED_DS_DIR=$WORK_DIR/Datasets/deep-stylometry/answerdotai-modernbert-base/no-padding/
CHECKPOINT_PATH=$PROJECT_ROOT/tmp/answerdotai-ModernBERT-base__halvest__pooling-li/step-step=23000.ckpt
LOGS_DIR=$PROJECT_ROOT/logs

# --------------------------------------------------------------------------------------

# CACHE_DIR=$SCRATCH/cache
NUM_PROC=16

# **************************************************************************************

mkdir -p "$LOGS_DIR" || true
if [[ -v CHECKPOINT_DIR ]]; then
    mkdir -p "$CHECKPOINT_DIR" || true
fi

if [[ $SLURM_JOB_ID != "" ]]; then
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "SLURM_NNODES: $SLURM_NNODES"
    echo "SLURM_NTASKS: $SLURM_NTASKS"
    echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
    echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$SLURM_GPUS_ON_NODE \
        "$PROJECT_ROOT/test.py" \
        --config_path "$CONFIG_PATH" \
        --processed_ds_dir "$PROCESSED_DS_DIR" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --logs_dir "$LOGS_DIR" \
        ${CACHE_DIR:+--cache_dir "$CACHE_DIR"} \
        ${NUM_PROC:+--num_proc "$NUM_PROC"}
else
    cmd=()
    cmd+=(python3 "$PROJECT_ROOT/test.py"
        --config_path "$CONFIG_PATH"
        --processed_ds_dir "$PROCESSED_DS_DIR"
        --logs_dir "$LOGS_DIR"
        --checkpoint_path "$CHECKPOINT_PATH")

    if [[ -v CACHE_DIR ]]; then
        mkdir -p "$CACHE_DIR" || true
        cmd+=(--cache_dir "$CACHE_DIR")
    fi

    if [[ -v NUM_PROC ]]; then
        cmd+=(--num_proc "$NUM_PROC")
    fi

    "${cmd[@]}"
fi
