#!/bin/bash
# scripts/collab_stats.sh
#
# Run collaboration_statistics.py — author-set size distributions, single-author
# rates, author-set recurrence, and cross-triplet Jaccard for HALvest-Contrastive.
#
# CPU-only (no GPU required).  Designed to run locally or inside a SLURM job
# via collab_stats.sbatch.
#
# Optional environment variables:
#   HALVEST_CACHE        HuggingFace cache directory (forwarded as --cache-dir)
#   COLLAB_OUTPUT_DIR    Where to write outputs  (default: <project_root>/stats_output)
#   COLLAB_CONFIGS       Space-separated list of HALvest configs to process
#                        (default: "base-2 base-4 base-6 base-8 base-10")
#   COLLAB_SPLIT         HuggingFace split to use  (default: train)
#   NUM_PROC             Worker count (not directly used by collaboration_statistics.py
#                        but kept for consistency; HF datasets.map() respects it)
#
# Any extra positional arguments are forwarded verbatim to collaboration_statistics.py.

set -euo pipefail

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR=${COLLAB_OUTPUT_DIR:-$PROJECT_ROOT/stats_output}
CONFIGS=${COLLAB_CONFIGS:-"base-2 base-4 base-6 base-8 base-10"}
SPLIT=${COLLAB_SPLIT:-valid}

mkdir -p "$OUTPUT_DIR"

# HuggingFace datasets: use spawn to avoid fork + CUDA incompatibility.
export DATASETS_MULTIPROCESS_CONTEXT=${DATASETS_MULTIPROCESS_CONTEXT:-spawn}

# One BLAS/OpenMP thread per worker process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------------------
# SLURM diagnostic info
# ---------------------------------------------------------------------------

if [[ "${SLURM_JOB_ID:-}" != "" ]]; then
    echo "SLURM_JOB_ID:        $SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST:  $SLURM_JOB_NODELIST"
    echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-?}"
    echo "OUTPUT_DIR:          $OUTPUT_DIR"
    echo "CONFIGS:             $CONFIGS"
    echo "SPLIT:               $SPLIT"
fi

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------

# shellcheck disable=SC2206
CONFIGS_ARRAY=($CONFIGS)

cmd=(
    python3 -m deep_stylometry.experiments.collaboration_statistics
    --output-dir "$OUTPUT_DIR"
    --configs "${CONFIGS_ARRAY[@]}"
    --split "$SPLIT"
)

if [[ -n "${HALVEST_CACHE:-}" ]]; then
    cmd+=(--cache-dir "$HALVEST_CACHE")
fi

# Forward any extra arguments
cmd+=("$@")

echo ""
echo "=== collaboration_statistics command ==="
echo "${cmd[*]}"
echo "========================================"

"${cmd[@]}"
