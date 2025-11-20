#!/bin/bash
#SBATCH --job-name=trim_to_test
#SBATCH --partition=big
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs

# ---- Env ----

# Load required modules
module purge
module load gnu12/12.2.0 openmpi4/4.1.4
module load cdo/2.1.1
module load nco/5.1.3

source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# test_subset and time_slice cannot both be activated

python -u S_land.py


