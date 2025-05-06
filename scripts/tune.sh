#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

CONFIG_PATH=$PROJECT_ROOT/configs/tune.yml
RAY_STORAGE_PATH=$PROJECT_ROOT/ray_logs

# --------------------------------------------------------------------------------------

# CACHE_DIR=$DATA_ROOT/responses/
# NUM_PROC=4
#

# **************************************************************************************

mkdir logs || true
mkdir -p "$RAY_STORAGE_PATH" || true

cmd=(python3 "$PROJECT_ROOT/tune.py"
    --config_path "$CONFIG_PATH"
    --ray_storage_path "$RAY_STORAGE_PATH")

if [[ -v CACHE_DIR ]]; then
    mkdir -p "$CACHE_DIR" || true
    cmd+=(--cache_dir "$CACHE_DIR")
fi

if [[ -v NUM_PROC ]]; then
    cmd+=(--num_proc "$NUM_PROC")
fi

"${cmd[@]}"
