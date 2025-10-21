#!/bin/bash
#SBATCH --partition=big
#SBATCH --job-name=init_training_zarrs
#SBATCH --output=logs/init_%A_%a.out
#SBATCH --error=logs/init_%A_%a.err
#SBATCH --array=0-20
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

module purge
module load gnu12/12.2.0 openmpi4/4.1.4
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

python -u make_empty_zarrs.py