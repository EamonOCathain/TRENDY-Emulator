#!/bin/bash
#SBATCH --job-name=standardisation
#SBATCH --partition=big
#SBATCH --array=0-90
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Activate conda
set +u
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# Unbuffered run so logs flush as it goes
python -u standardisation.py