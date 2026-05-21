#!/bin/bash

#SBATCH --job-name=collab-stats-deep-stylometry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # 8 cores for HF datasets.map() workers
#SBATCH --mem=64G                  # HALvest Arrow dataset fits comfortably
#SBATCH --partition=cpu_homogen    # CPU-only: no GPU needed for this analysis
#SBATCH --time=2:00:00             # collaboration stats are faster than full stats
#SBATCH --output=logs/%x-%j-%t.log

mkdir -p logs

module purge
module load slurm 2>/dev/null || true  # not available on all partitions; non-fatal
source "$HOME/.bashrc"

conda activate deep-stylometry
if [[ "$CONDA_DEFAULT_ENV" != "deep-stylometry" ]]; then
    echo "Error: Failed to activate conda environment 'deep-stylometry'" >&2
    exit 1
fi

# Optional: point to a warm HF cache to skip re-downloading HALvest.
# export HALVEST_CACHE=$WORK_DIR/hf_cache

# Optional: override defaults.
# export COLLAB_OUTPUT_DIR=$WORK_DIR/stats_output
# export COLLAB_CONFIGS="base-2 base-4 base-6 base-8 base-10"
# export COLLAB_SPLIT=train

./scripts/collab_stats.sh "$@"
