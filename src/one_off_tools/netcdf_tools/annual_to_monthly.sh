#!/bin/bash
#SBATCH --job-name=chunker
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
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
python -u annual_to_monthly.py \
    --in_nc /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/counter_factuals/global/DATA/cLitter/ENSMEAN/ENSMEAN_S3_cLitter.nc \
    --out_nc /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/counter_factuals/global/DATA/cLitter/ENSMEAN/ENSMEAN_S3_cLitter_monthly.nc