#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=fill_pot_rad
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-95

set -euo pipefail
mkdir -p logs

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLOSC_NTHREADS=1
export VAR_WORKERS=4

source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

python -u fill_potential_rad.py 
