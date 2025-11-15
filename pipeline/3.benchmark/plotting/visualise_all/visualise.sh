#!/bin/bash
#SBATCH --job-name=analyse_metrics
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-134%60
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs

# ---- Env ----
module purge
set +u
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base
set -u

# test_subset and time_slice cannot both be activated

python -u visualise.py \
  --job_name base_model/no_carry/test_early \
  --preds_dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/base_model/no_carry \
  --labels_dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference \
  --plot_dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/plots \
  --cache_dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/analysis/metrics_cache \
  --scenario S3 \
  --time_slice '1901-01-01,1918-12-31'