#!/bin/bash

#SBATCH --job-name=zero-shot-e5-halvest
#SBATCH --partition=almanach
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j-%t.log

module purge
module load slurm
source $HOME/.bashrc

export WANDB_CACHE_DIR="$WORK_DIR/wandb_cache"
export WANDB_DATA_DIR="$WORK_DIR/wandb_data"
export WANDB_CONFIG_DIR="$WORK_DIR/wandb_config"

conda activate deep-stylometry
if [[ "$CONDA_DEFAULT_ENV" != "deep-stylometry" ]]; then
    echo "Error: Failed to activate environment"
    exit 1
fi

./scripts/test_zero_shot.sh "$@"
