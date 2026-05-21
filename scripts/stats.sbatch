#!/bin/bash

#SBATCH --job-name=stats-deep-stylometry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # 8 cores → HF datasets.map() workers
#SBATCH --mem=64G                  # HALvest Arrow dataset fits comfortably
#SBATCH --partition=cpu_homogen    # switch to cpu_devl for fast scheduling
#SBATCH --time=4:00:00             # HALvest entropy loop ~2–3 h on first run
#SBATCH --output=logs/%x-%j-%t.log

mkdir -p logs

module purge
module load slurm 2>/dev/null || true # not available on all partitions; non-fatal
source "$HOME/.bashrc"

conda activate deep-stylometry
if [[ "$CONDA_DEFAULT_ENV" != "deep-stylometry" ]]; then
    echo "Error: Failed to activate conda environment 'deep-stylometry'" >&2
    exit 1
fi

# PAN 2019 archive — set here or export before sbatch
export PAN19_ZIP=${PAN19_ZIP:-$PWD/data/pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23.zip}

# Optional: point to a warm HF cache to skip re-downloading HALvest
# export HALVEST_CACHE=$WORK_DIR/hf_cache

./scripts/stats.sh "$@"
