#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=consol_infer
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1

mkdir -p logs

# (Optional) keep threaded libs from oversubscribing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# Run: pass time-res as a flag; no need to pass the array ID
python -u consolidate.py