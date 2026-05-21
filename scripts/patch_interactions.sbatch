#!/bin/bash

#SBATCH --job-name=patch-interactions-deep-stylometry
#SBATCH --partition=almanach
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j-%t.log

mkdir -p logs

module purge
module load slurm 2>/dev/null || true
source "$HOME/.bashrc"

conda activate deep-stylometry
if [[ "$CONDA_DEFAULT_ENV" != "deep-stylometry" ]]; then
    echo "Error: Failed to activate conda environment 'deep-stylometry'" >&2
    exit 1
fi

export TOKENIZERS_PARALLELISM=false

./scripts/patch_interactions.sh "$@"
