#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/predictions/%x_%A_%a.out
#SBATCH --error=logs/predictions/%x_%A_%a.err

set -euo pipefail

# Load config
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SOURCE_DIR}/config.sh"

# Env
module purge
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

mkdir -p "${PREDICT_LOG_DIR}"

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u predict.py \
  --job_name "${JOB_NAME}" \
  --out_dir "${OUT_DIR}" \
  --scenario "${SCENARIO}" \
  --forcing_dir "${FORCING_DIR}" \
  --weights "${WEIGHTS}" \
  --store_period "${STORE_PERIOD}" \
  --write_period "${WRITE_PERIOD}" \
  --shards "${SLURM_ARRAY_TASK_COUNT}" \
  --shard_id "${SLURM_ARRAY_TASK_ID}" \
  --nudge_lambda "${NUDGE_LAMBDA}" \
  --nudge_mode "${NUDGE_MODE}" \
  --carry_forward_states "${CARRY_FORWARD_STATES}" \
  --sequential_months "${SEQUENTIAL_MONTHS}" \
  --exclude_vars "${EXCLUDE_VARS:-}" \
  --tl_vars ${TL_VARS} \
  --tl_initial_state "${TL_INITIAL_STATE}"

# tip: add --overwrite_data above if you want to re-run tiles