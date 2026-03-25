#!/bin/bash

#SBATCH --job-name=test-deep-stylometry               # Job name
#SBATCH --account=grandchallenge1
#SBATCH --nodes=1                                    # Run all processes on a single node
#SBATCH --ntasks=1                                      # Run a single task
#SBATCH --mem=256G                                     # Job memory request
#SBATCH --gres=gpu:1                             # Request 1 GPU
#SBATCH --cpus-per-task=16                              # Number of CPU cores per task
#SBATCH --time=2-00:00:00                               # Time limit hrs:min:sec
#SBATCH --output=logs/%x-%j-%t.log                         # Standard output and error log. %x denotes the job name, %j the jobid.

module purge
module load slurm
source $HOME/.bashrc

export WANDB_CACHE_DIR="$WORK_DIR/wandb_cache"
export WANDB_DATA_DIR="$WORK_DIR/wandb_data"
export WANDB_CONFIG_DIR="$WORK_DIR/wandb_config"
export TORCHELASTIC_ERROR_FILE="$PWD/logs/torch_error_${SLURM_JOB_ID}.json"

conda activate deep-stylometry
if [[ "$CONDA_DEFAULT_ENV" != "deep-stylometry" ]]; then
    echo "Error: Failed to activate environment"
    exit 1
fi

export DATASETS_MULTIPROCESS_CONTEXT=spawn

./scripts/test.sh "$@"
