#!/bin/sh

#SBATCH --job-name=deep_strylometry_train                # Job name
#SBATCH --nodes=1                                       # Run all processes on a single node
#SBATCH --ntasks=2                                      # Run a single task
#SBATCH --ntasks-per-node=2                                # Number of tasks per node
#SBATCH --partition=almanach                            # GPU parition
#SBATCH --gres=gpu:rtx8000:2                                    # Request 1 GPU
#SBATCH --cpus-per-task=16                              # Number of CPU cores per task
#SBATCH --time=2-00:00:00                               # Time limit hrs:min:sec
#SBATCH --output=logs/%x_%j.log                         # Standard output and error log. %x denotes the job name, %j the jobid.

module purge

source /home/$USER/.bashrc
conda activate deep_stylometry

mkdir logs || true

./scripts/train.sh
