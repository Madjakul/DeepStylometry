#!/bin/sh

#SBATCH --job-name=tune-pli-deep-stylometry         # Job name
#SBATCH --account=grandchallenge1
#SBATCH --nodes=1                                    # Single node
#SBATCH --ntasks=1                                   # Single task
#SBATCH --mem=64G                                    # Memory (1 GPU, smaller than train)
#SBATCH --gres=gpu:1                                 # 1 GPU per trial
#SBATCH --cpus-per-task=10                           # CPU cores for dataloading
#SBATCH --time=1-00:00:00                            # 1 day for 30 trials x 10k steps
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

export DATASETS_MULTIPROCESS_CONTEXT=spawn

./scripts/tune.sh "$@"
