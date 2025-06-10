#!/bin/sh

#SBATCH --job-name=deep_strylometry_train                # Job name
#SBATCH --nodes=1                                       # Run all processes on a single node
#SBATCH --ntasks=3                                      # Run a single task
#SBATCH --partition=almanach                            # GPU parition
#SBATCH --gres=gpu:rtx8000:3                            # Request 1 GPU
#SBATCH --cpus-per-task=8                              # Number of CPU cores per task
#SBATCH --time=2-00:00:00                               # Time limit hrs:min:sec
#SBATCH --output=logs/%x_%j_%t.log                         # Standard output and error log. %x denotes the job name, %j the jobid.

module purge

source /home/$USER/.bashrc
conda activate deep_stylometry

mkdir logs || true

export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 # (e.g., 1MB)
# export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export DATASETS_MULTIPROCESS_CONTEXT=spawn

./scripts/train.sh
