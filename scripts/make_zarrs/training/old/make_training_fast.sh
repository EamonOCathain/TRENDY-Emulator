#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=mk_tr_fst
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-20
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

set -euo pipefail
mkdir -p logs

# Threading hygiene
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLOSC_NTHREADS=$SLURM_CPUS_PER_TASK

# Activate env
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# Run writer. Keep --validate to do the final full-store NaN scan.
python -u make_training_fast.py --validate --daily_files_mode twenty