#!/bin/bash
#SBATCH --job-name=mass_balance
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-99
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module purge
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p logs

export PYTHONUNBUFFERED=1
echo "[INFO] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

python /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/training/mass_balance_labels/mass_balance_labels.py \
  --split train \
  --n_shards 100 \
  --shard_id "${SLURM_ARRAY_TASK_ID}" \
  --num_workers 8 \
  --batch_size 1 \
  --out_dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/training/mass_balance_labels/shards