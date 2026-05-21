#!/bin/bash
# scripts/patch_interactions.sh
#
# Run patch_interactions.py — patch-level interaction visualiser.
#
# Usage (local):
#   ./scripts/patch_interactions.sh [CONFIG_PATH] [CHECKPOINT_PATH]
#
# Usage (SLURM):
#   sbatch scripts/patch_interactions.sbatch [CONFIG_PATH] [CHECKPOINT_PATH]
#
# Positional arguments:
#   $1   Config YAML path    (default: configs/test_pli_learned.yml)
#   $2   Checkpoint path     (default: last PLI-learned checkpoint in tmp/)
#
# Any arguments after '--' are forwarded verbatim to patch_interactions.py.
#
# Optional environment variables:
#   DATASET       Dataset to use: 'pan19' or 'halvest' (default: pan19)
#   SUBSET        HALvest-Contrastive subset — ignored when DATASET=pan19 (default: base-10)
#   SEED          Random seed for PAN19 triplet sampling (default: 42)
#   PAN19_LANGUAGE  PAN19 language filter (default: en)
#   N_SAMPLES     Number of samples to analyse (default: 200)
#   N_VIZ         Number of examples to render in HTML (default: 20 for pan19, 50 for halvest)
#   OUTPUT_HTML   Output HTML path (default: <project_root>/figures/interactions_patch_pan19.html)
#   PAN19_ZIP     Path to PAN19 ZIP archive (required when DATASET=pan19; or set env var)

set -euo pipefail

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

CONFIG_PATH=${1:-$PROJECT_ROOT/configs/test_pli_learned.yml}
CHECKPOINT_PATH=${2:-$PROJECT_ROOT/tmp/answerdotai-modernbert-base__halvest__pooling-pli-learned-n3__skip_list-true/last.ckpt}

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

DATASET=${DATASET:-pan19}
SUBSET=${SUBSET:-base-10}
SEED=${SEED:-42}
PAN19_LANGUAGE=${PAN19_LANGUAGE:-en}
export PAN19_ZIP=${PAN19_ZIP:-$DATA_ROOT/pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23.zip}
N_SAMPLES=${N_SAMPLES:-200}

# Default n_viz and output path depend on dataset
if [[ "$DATASET" == "pan19" ]]; then
    N_VIZ=${N_VIZ:-20}
    OUTPUT_HTML=${OUTPUT_HTML:-$PROJECT_ROOT/figures/interactions_patch_pan19.html}
else
    N_VIZ=${N_VIZ:-50}
    OUTPUT_HTML=${OUTPUT_HTML:-$PROJECT_ROOT/figures/interactions_patch_${SUBSET}.html}
fi

mkdir -p "$(dirname "$OUTPUT_HTML")"

# **************************************************************************************

if [[ "${SLURM_JOB_ID:-}" != "" ]]; then
    echo "SLURM_JOB_ID:        $SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST:  $SLURM_JOB_NODELIST"
    echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES:-?}"
    echo "CONFIG_PATH:         $CONFIG_PATH"
    echo "CHECKPOINT_PATH:     $CHECKPOINT_PATH"
    echo "DATASET:             $DATASET"
    echo "SUBSET:              $SUBSET (ignored if DATASET=pan19)"
    echo "SEED:                $SEED"
    echo "PAN19_LANGUAGE:      $PAN19_LANGUAGE"
    echo "PAN19_ZIP:           $PAN19_ZIP"
    echo "N_SAMPLES:           $N_SAMPLES"
    echo "N_VIZ:               $N_VIZ"
    echo "OUTPUT_HTML:         $OUTPUT_HTML"
fi

cmd=(
    python3 -m deep_stylometry.experiments.patch_interactions
    --config_path "$CONFIG_PATH"
    --checkpoint_path "$CHECKPOINT_PATH"
    --dataset "$DATASET"
    --subset "$SUBSET"
    --seed "$SEED"
    --pan19_language "$PAN19_LANGUAGE"
    --n_samples "$N_SAMPLES"
    --n_viz "$N_VIZ"
    --output_html "$OUTPUT_HTML"
)

# Forward PAN19_ZIP explicitly so the printed command is self-documenting.
cmd+=(--pan19_zip "$PAN19_ZIP")

cmd+=("${EXTRA_ARGS[@]}")

echo ""
echo "=== patch_interactions command ==="
echo "${cmd[*]}"
echo "=================================="
"${cmd[@]}"
