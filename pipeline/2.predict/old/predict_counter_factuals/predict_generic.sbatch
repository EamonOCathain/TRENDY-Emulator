#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --array=0-49
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --chdir=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/2.predict/predict_counter_factuals

set -euo pipefail
mkdir -p logs

# ---- Load config (defaults). Anything pre-set in the environment will override these. ----
source ./predict_config.sh

# (environment hygiene)
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Avoid nounset issues in conda MKL hooks on this cluster
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=INTEL

# ---- Conda env ----
module purge
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# ---- Run ----
python -u "${PYTHON}" \
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
  --exclude_vars "${EXCLUDE_VARS}" \
  --device "${DEVICE}" \
  --ilamb_dir_global "${ILAMB_DIR_GLOBAL}" \
  --number_tiles "${NUMBER_TILES}" \
  --forcing_offsets "${FORCING_OFFSETS}" \
  --export_nc