#!/bin/bash
#SBATCH --job-name=seasonality_fill
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --partition=big

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------
set -euo pipefail

echo "[INFO] Job started on $(hostname) at $(date)"
echo "[INFO] SLURM job ID: $SLURM_JOB_ID"

module purge
module load gnu12/12.2.0 openmpi4/4.1.4
module load cdo/2.1.1
module load nco/5.1.3

# Optional: for some clusters you need the right Python module too
# module load python/3.10

# Activate your conda environment
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# ---------------------------------------------------------------------
# Run script
# ---------------------------------------------------------------------
python -u fill_with_seasonality.py

echo "[INFO] Job finished at $(date)"