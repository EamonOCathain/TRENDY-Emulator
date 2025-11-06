#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=check_nans
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=3-00:00:00
#SBATCH --mem=8G
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
python check_nans_zarr.py \
    /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/training_new/val \
    --verbose