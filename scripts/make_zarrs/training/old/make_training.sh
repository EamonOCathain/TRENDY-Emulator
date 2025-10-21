#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=mk_trning
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-20
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

mkdir -p logs

# (Optional) keep threaded libs from oversubscribing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLOSC_NTHREADS=$SLURM_CPUS_PER_TASK

source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# Run: pass time-res as a flag; no need to pass the array ID
python -u make_training.py --daily_files_mode twenty