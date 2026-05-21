#!/bin/bash

#SBATCH --job-name=decorrelation-deep-stylometry
#SBATCH --partition=almanach
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G          # fine-tuned model + E5 model fit in CPU RAM together
#SBATCH --gres=gpu:1        # decorrelation forces devices=1 internally
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00      # test pass over full corpus + E5 predict
#SBATCH --output=logs/%x-%j-%t.log

mkdir -p logs

module purge
module load slurm 2>/dev/null || true # not available on all partitions;
source "$HOME/.bashrc"

export WANDB_CACHE_DIR="$WORK_DIR/wandb_cache"
export WANDB_DATA_DIR="$WORK_DIR/wandb_data"
export WANDB_CONFIG_DIR="$WORK_DIR/wandb_config"

conda activate deep-stylometry
if [[ "$CONDA_DEFAULT_ENV" != "deep-stylometry" ]]; then
    echo "Error: Failed to activate conda environment 'deep-stylometry'" >&2
    exit 1
fi

# HF datasets: spawn avoids fork-after-CUDA errors during map().
export DATASETS_MULTIPROCESS_CONTEXT=spawn
export TOKENIZERS_PARALLELISM=false

./scripts/decorrelation.sh "$@"
