#!/bin/bash
#SBATCH --job-name=export_nc
#SBATCH --partition=work
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=predictions/logs/netcdfs/%x_%A_%a.out
#SBATCH --error=predictions/logs/netcdfs/%x_%A_%a.err

set -euo pipefail

# Load config
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SOURCE_DIR}/config.sh"

# Env
module purge
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

mkdir -p "${EXPORT_LOG_DIR}"

# Export-only pass: no model/weights; leader will consolidate and export.
python -u predict.py \
  --job_name "${JOB_NAME}" \
  --out_dir "${OUT_DIR}" \
  --scenario "${SCENARIO}" \
  --forcing_dir "${FORCING_DIR}" \
  --store_period "${STORE_PERIOD}" \
  --write_period "${WRITE_PERIOD}" \
  --shards "${SLURM_ARRAY_TASK_COUNT}" \
  --shard_id "${SLURM_ARRAY_TASK_ID}" \
  --weights "${WEIGHTS}" \
  --export_nc_only \
