#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=finalize_t6
#SBATCH --output=logs/finalize_t6.out
#SBATCH --error=logs/finalize_t6.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-20

set -euo pipefail
mkdir -p logs

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# Finalize only (task 6), with validation
export FINALIZE=1
python -u make_training_tiles.py --daily_files_mode twenty --validate