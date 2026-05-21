#!/bin/bash
# scripts/decorrelation.sh
#
# Run semantic_decorrelation.py — semantic vs. stylistic cosine similarity
# decorrelation analysis for a given fine-tuned checkpoint.
#
# Usage (local):
#   ./scripts/decorrelation.sh [CONFIG_PATH] [CHECKPOINT_PATH]
#
# Usage (SLURM):
#   sbatch scripts/decorrelation.sbatch [CONFIG_PATH] [CHECKPOINT_PATH]
#
# Positional arguments:
#   $1   Config YAML path   (default: configs/decorrelation_pli_learned.yml)
#   $2   Checkpoint path    (default: last PLI-learned checkpoint in tmp/)
#
# Any arguments after '--' are forwarded verbatim to semantic_decorrelation.py.
#
# The decorrelation-specific settings (subset, sample_size, output_dir, …) are
# read from the YAML's  decorrelation:  section.  Pass explicit CLI args after
# '--' to override individual values without touching the config.
#
# Optional environment variables:
#   CACHE_DIR     HuggingFace cache directory (forwarded as --hf-cache-dir)
#   NUM_PROC      Worker count for HF datasets.map() (default: 16)

set -euo pipefail

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify

# ************************** Customizable Arguments ************************************

CONFIG_PATH=${1:-$PROJECT_ROOT/configs/decorrelation.yml}
CHECKPOINT_PATH=${2:-$PROJECT_ROOT/tmp/answerdotai-modernbert-base__halvest__pooling-mean/last.ckpt}
PROCESSED_DS_DIR=$SCRATCH/Datasets/deep-stylometry/answerdotai-modernbert-base/no-padding/

# Collect extra args: everything after the '--' separator
EXTRA_ARGS=()
shift 2 2>/dev/null || true
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
NUM_PROC=${NUM_PROC:-${SLURM_CPUS_PER_TASK:-16}}

# **************************************************************************************

# Decorrelation always runs on a single GPU — no torchrun needed.
if [[ "${SLURM_JOB_ID:-}" != "" ]]; then
    echo "SLURM_JOB_ID:        $SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST:  $SLURM_JOB_NODELIST"
    echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES:-?}"
    echo "CONFIG_PATH:         $CONFIG_PATH"
    echo "CHECKPOINT_PATH:     $CHECKPOINT_PATH"
    echo "NUM_PROC:            $NUM_PROC"
fi

cmd=(
    python3 -m deep_stylometry.experiments.semantic_decorrelation
    --checkpoint "$CHECKPOINT_PATH"
    --config "$CONFIG_PATH"
    --processed-ds-dir "$PROCESSED_DS_DIR"
    --device cuda
    --num-proc "$NUM_PROC"
)

if [[ -v CACHE_DIR ]]; then
    mkdir -p "$CACHE_DIR" || true
    cmd+=(--hf-cache-dir "$CACHE_DIR")
fi

cmd+=("${EXTRA_ARGS[@]}")

echo ""
echo "=== semantic_decorrelation command ==="
echo "${cmd[*]}"
echo "======================================"
"${cmd[@]}"
