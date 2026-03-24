#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify

# ************************** Customizable Arguments ************************************

# $1=config_path (default: train_pli_learned as base)
CONFIG_PATH=${1:-$PROJECT_ROOT/configs/train_pli_learned.yml}
PROCESSED_DS_DIR=$WORK_DIR/Datasets/deep-stylometry/answerdotai-modernbert-base/no-padding/
LOGS_DIR=$PROJECT_ROOT/logs
OUTPUT_PATH=$PROJECT_ROOT/configs/best_pli_params.yml

# --------------------------------------------------------------------------------------

N_TRIALS=30
MAX_STEPS=10000
STUDY_NAME="pli_search"
NUM_PROC=10

# Optional: uncomment to persist study across runs
# STORAGE="sqlite:///$PROJECT_ROOT/logs/pli_optuna.db"

# **************************************************************************************

mkdir -p "$LOGS_DIR" || true
mkdir -p "$(dirname "$OUTPUT_PATH")" || true

if [[ $SLURM_JOB_ID != "" ]]; then
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

    python3 "$PROJECT_ROOT/tune.py" \
        --config_path "$CONFIG_PATH" \
        --processed_ds_dir "$PROCESSED_DS_DIR" \
        --logs_dir "$LOGS_DIR" \
        --n_trials "$N_TRIALS" \
        --max_steps "$MAX_STEPS" \
        --output_path "$OUTPUT_PATH" \
        --study_name "$STUDY_NAME" \
        --num_proc "$NUM_PROC" \
        ${STORAGE:+--storage "$STORAGE"}
else
    cmd=(python3 "$PROJECT_ROOT/tune.py"
        --config_path "$CONFIG_PATH"
        --processed_ds_dir "$PROCESSED_DS_DIR"
        --logs_dir "$LOGS_DIR"
        --n_trials "$N_TRIALS"
        --max_steps "$MAX_STEPS"
        --output_path "$OUTPUT_PATH"
        --study_name "$STUDY_NAME"
        --num_proc "$NUM_PROC")

    if [[ -v STORAGE ]]; then
        cmd+=(--storage "$STORAGE")
    fi

    "${cmd[@]}"
fi
