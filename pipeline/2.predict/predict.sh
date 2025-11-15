#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=50G
#SBATCH --array=0-7
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=gpu:1

# ---- USER PARAMS ----
: "${JOB_NAME:=32_years_1982_2018}"
: "${CARRY_FORWARD_STATES=True}"
: "${SEQUENTIAL_MONTHS=True}"
: "${SCENARIO:=S3}"
# Device
: "${DEVICE:=cuda}"
: "${ILAMB_DIR_GLOBAL:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark}"
: "${NUMBER_TILES:=8}"
# Weights
: "${WEIGHTS:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/checkpoints/carry/32_year/checkpoints/best.pt}"

# Carrying and Nudging
: "${NUDGE_LAMBDA:=0}" 
: "${NUDGE_MODE=none}"

# Params You Rarely Change
: "${FORCING_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference}"
: "${OUT_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/transfer_learn}"

: "${STORE_PERIOD:=1982-01-01:2018-12-31}"
: "${WRITE_PERIOD:=1982-01-01:2018-12-31}"

# ---- Env ----
module purge
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate trendy-gpu
mkdir -p logs

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/2.predict/predict.py \
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
  --export_nc_only

# --forcing_offsets "daily:tmp=+5,daily:tmin=+5,daily:tmax=+5,annual:tmp_rolling_mean=+5,annual:tmin_rolling_mean=+5,annual:tmax_rolling_mean=+5"
#   --overwrite_data