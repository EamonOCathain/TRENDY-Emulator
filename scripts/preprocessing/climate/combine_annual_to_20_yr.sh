#!/bin/bash
#SBATCH --partition=big
#SBATCH --job-name=climate_processor
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-174%89
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

# Load required modules
module purge
module load gnu12/12.2.0 openmpi4/4.1.4
module load cdo/2.1.1
module load nco/5.1.3

# Activate your conda environment
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# Run the Python script with SLURM task ID as argument
python -u combine_annual_to_20_yr.py 