#!/bin/sh

#SBATCH --job-name=deep_strylometry_preprocess    # Job name
#SBATCH --nodes=1                                       # Run all processes on a single node
#SBATCH --ntasks=1                                      # Run a single task	
#SBATCH --partition=cpu_devel                         # CPU partition
#SBATCH --time=2-00:00:00                               # Time limit hrs:min:sec
#SBATCH --output=logs/%x_%j.log                         # Standard output and error log. %x denotes the job name, %j the jobid.
#SBATCH --exclusive                                     # Request exclusive node

module purge

source /home/$USER/.bashrc
conda activate halvesting_contrastive

mkdir logs || true

./scripts/preprocess.sh
