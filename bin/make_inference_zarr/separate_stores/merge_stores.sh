#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=infer_sep
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=3-00:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1

# Activate your conda environment
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# Run the Python script with SLURM task ID as argument
python -u merge_stores.py