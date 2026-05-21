#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

# First positional argument overrides the config path
CONFIG_PATH=${1:-$PROJECT_ROOT/configs/mechanistic.yml}
# Second positional argument selects the phase(s) to run.
# Accepts any value accepted by run_phase.py --phase: 0, 1a, 1b, 2, 3, 4, all.
# Multiple phases can be passed as a space-separated quoted string, e.g. "0 1a".
PHASE=${2:-"2 4"} # "0 1b 2 3 4"}
LOGS_DIR=$PROJECT_ROOT/logs

# Collect extra args: everything after the first two positional args
EXTRA_ARGS=()
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

# --------------------------------------------------------------------------------------

# **************************************************************************************

mkdir -p "$LOGS_DIR" || true

if [[ $SLURM_JOB_ID != "" ]]; then
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "SLURM_NNODES: $SLURM_NNODES"
    echo "SLURM_NTASKS: $SLURM_NTASKS"
    echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
    echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

    python3 -m deep_stylometry.experiments.mechanistic.run_phase \
        --config "$CONFIG_PATH" \
        --phase $PHASE \
        --resume \
        "${EXTRA_ARGS[@]}"
else
    cmd=(python3 -m deep_stylometry.experiments.mechanistic.run_phase
        --config "$CONFIG_PATH"
        --phase $PHASE)

    cmd+=("${EXTRA_ARGS[@]}")

    "${cmd[@]}"
fi
