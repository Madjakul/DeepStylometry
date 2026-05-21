#!/bin/bash
# scripts/stats.sh
#
# Run dataset_statistics.py (HALvest + PAN 2019) and produce JSON, LaTeX tables,
# and PDF figures in OUTPUT_DIR.
#
# Designed to run on a CPU-only SLURM partition (cpu_devl / cpu_homogen) or
# locally.  When executed inside a SLURM job the script reads SLURM_CPUS_PER_TASK
# automatically; set NUM_PROC to override.
#
# Required environment variables (set before calling / in stats.sbatch):
#   PAN19_ZIP          Path to the PAN 2019 ZIP archive  (preferred)
#   PAN19_ROOT         Path to the extracted directory    (fallback / tests)
#
# Optional environment variables:
#   HALVEST_CACHE      HuggingFace cache directory (default: HF_HOME)
#   STATS_OUTPUT_DIR   Where to write outputs (default: <project_root>/stats_output)
#   NUM_PROC           Worker count for HF datasets.map() (default: SLURM_CPUS_PER_TASK or 4)
#   SKIP_HALVEST       Set to "1" to skip HALvest (fast PAN-only run)
#   SKIP_PAN19         Set to "1" to skip PAN 2019
#
# Any extra positional arguments are forwarded verbatim to dataset_statistics.py.

set -euo pipefail

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..

# ---------------------------------------------------------------------------
# Parallelism configuration
# ---------------------------------------------------------------------------

# Use as many cores as SLURM allocated; cap at 16 to avoid thrashing on nodes
# shared with other users.  A local run defaults to 4.
NUM_PROC=${NUM_PROC:-${SLURM_CPUS_PER_TASK:-4}}
NUM_PROC=$((NUM_PROC > 16 ? 16 : NUM_PROC))

# HuggingFace datasets: use spawn to avoid fork + CUDA incompatibility.
export DATASETS_MULTIPROCESS_CONTEXT=${DATASETS_MULTIPROCESS_CONTEXT:-spawn}

# One BLAS/OpenMP thread per worker process; dataset workers do the parallelism.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# HF tokenizers is not used in stats, but silence the parallelism warning.
export TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR=${STATS_OUTPUT_DIR:-$PROJECT_ROOT/stats_output}
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# SLURM diagnostic info (logged when running inside a job)
# ---------------------------------------------------------------------------

if [[ "${SLURM_JOB_ID:-}" != "" ]]; then
    echo "SLURM_JOB_ID:        $SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST:  $SLURM_JOB_NODELIST"
    echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-?}"
    echo "NUM_PROC (effective): $NUM_PROC"
    echo "OUTPUT_DIR:          $OUTPUT_DIR"
fi

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------

cmd=(
    python3 -m deep_stylometry.experiments.dataset_statistics
    --output-dir "$OUTPUT_DIR"
)

if [[ "${SKIP_HALVEST:-0}" == "1" ]]; then
    cmd+=(--skip-halvest)
elif [[ -n "${HALVEST_CACHE:-}" ]]; then
    cmd+=(--halvest-cache "$HALVEST_CACHE")
fi

if [[ "${SKIP_PAN19:-0}" == "1" ]]; then
    cmd+=(--skip-pan19)
elif [[ -n "${PAN19_ZIP:-}" ]]; then
    cmd+=(--pan19-zip "$PAN19_ZIP")
elif [[ -n "${PAN19_ROOT:-}" ]]; then
    cmd+=(--pan19-root "$PAN19_ROOT")
fi

# Forward any extra arguments (e.g. --pan19-language es)
cmd+=("$@")

echo ""
echo "=== dataset_statistics command ==="
echo "${cmd[*]}"
echo "=================================="

"${cmd[@]}"
