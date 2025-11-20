#!/bin/bash
#SBATCH --job-name=analyse_metrics
#SBATCH --partition=big
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4

set -euo pipefail
mkdir -p logs

# ---- Env ----
module purge
set +u
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
# Use an env that has numpy/xarray/dask/matplotlib
conda activate base
set -u

# ---- Run: one figure per array task ----
python -u all_metrics.py \
  --dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/transfer_learn/TL_32_year_1982_2018/netcdf/S3 \
  --out_dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/analysis/metrics/TL_Emulator_1982_2018/S3/ \
  --start_year 1982 \
  --end_year 2018 