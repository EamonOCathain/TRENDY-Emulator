#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=infer_all
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-11
#SBATCH --time=3-00:00:00
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8

# Activate your conda environment
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

export BLOSC_NTHREADS=$SLURM_CPUS_PER_TASK

# Run the Python script with SLURM task ID as argument
python -u make_inference.py