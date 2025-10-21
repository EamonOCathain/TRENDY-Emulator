#!/bin/bash
#SBATCH --partition=big
#SBATCH --job-name=infer_write_annual
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-31
#SBATCH --time=3-00:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1

mkdir -p logs

# Prevent threaded libs from oversubscribing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# Change to your target directory
cd /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/make_zarrs/inference/temporary_stores/ || exit 1

# Run: pass time-res as a flag; no need to pass the array ID
python -u temp_stores_inference.py --time-res annual