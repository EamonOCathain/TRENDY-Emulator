#!/bin/bash
#SBATCH --partition=big
#SBATCH --job-name=rechunk
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-20  # adjust size after seeing how many .zarr you have

set -euo pipefail
mkdir -p logs

# Activate your env
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# I/O roots (edit these)
IN_DIR="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/training_rechunked_for_carry"
OUT_DIR="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/training_rechunked_for_carry_70"
TMP_DIR="/scratch/$USER/rechunk_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$TMP_DIR"

# Optional overrides for the sharding helper:
# export SHARD_STRATEGY=rr     # or "block"
# export SHARD_COUNT=          # force total shard count
# export SHARD_ID=             # force this shard id (0-based)
# export ONLY_INDEX=           # process exactly one item by global index

python -u rechunk_zarrs_for_carry.py \
  "$IN_DIR" "$OUT_DIR" \
  --tmp "$TMP_DIR" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --max-mem 8GB \
  --loc 70 \
  --scenario 1 \
  --time-daily 365 \
  --time-monthly 12 \
  --time-annual 1