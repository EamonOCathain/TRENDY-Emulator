#!/bin/bash
#SBATCH --job-name=take_trends
#SBATCH --partition=work
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --array=0-5
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

python -u take_trend.py \
  --in_dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S3 \
  --out_dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/trends/DATA