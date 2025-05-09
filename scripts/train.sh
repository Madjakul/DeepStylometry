#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

CONFIG_PATH=$PROJECT_ROOT/configs/preprocess.yml
LOGS_DIR=$PROJECT_ROOT/logs

# --------------------------------------------------------------------------------------

# CACHE_DIR=$DATA_ROOT/responses/
# CHECKPOINT_DIR=$DATA_ROOT/tmp/checkpoints/
# NUM_PROC=4
#

# **************************************************************************************

cmd=(python3 "$PROJECT_ROOT/train.py"
    --config_path "$CONFIG_PATH"
    --logs_dir "$LOGS_DIR")

if [[ -v CACHE_DIR ]]; then
    mkdir -p "$CACHE_DIR" || true
    cmd+=(--cache_dir "$CACHE_DIR")
fi

if [[ -v CHECKPOINT_DIR ]]; then
    mkdir -p "$CHECKPOINT_DIR" || true
    cmd+=(--checkpoint_dir "$CHECKPOINT_DIR")
fi

if [[ -v NUM_PROC ]]; then
    cmd+=(--num_proc "$NUM_PROC")
fi

"${cmd[@]}"
