#!/bin/bash
#SBATCH --job-name=exp_co2_121
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=24G
#SBATCH --array=0-7
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=gpu:1

# ---- USER PARAMS ----
: "${JOB_NAME:=counter_factuals/co2_offset_121.54/S3}"
# Scenario
: "${SCENARIO:=S3}"

# Weights
: "${WEIGHTS:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/1.train/runs/saved_checkpoints/base_model/base_model_new_loss/checkpoints/best.pt}"

# Carrying and Nudging
: "${CARRY_FORWARD_STATES=False}"
: "${SEQUENTIAL_MONTHS=False}"
: "${NUDGE_LAMBDA:=0}" 
: "${NUDGE_MODE=none}"

# Params You Rarely Change
: "${FORCING_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference}"
: "${OUT_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/}"

: "${STORE_PERIOD:=1901-01-01:2023-12-31}"
: "${WRITE_PERIOD:=1901-01-01:2023-12-31}"

# Exclude Variables
# "${EXCLUDE_VARS:="pre_rolling_mean,tmp_rolling_mean,spfh_rolling_mean,tmax_rolling_mean,tmin_rolling_mean"}"

# ---- Env ----
module purge
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate trendy-gpu
mkdir -p logs

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
  --forcing_offsets "annual:co2=121.54" \
  --export_nc_only

  