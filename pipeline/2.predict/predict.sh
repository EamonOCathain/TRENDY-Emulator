#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=24G
#SBATCH --array=0-7
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err


# ---- USER PARAMS ----
: "${JOB_NAME:=no_rolling_mean/no_carry_S3}"

# Weights
: "${WEIGHTS:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/a_pipeline/1.train/runs/saved_checkpoints/base_model/new_loss/no_rolling_mean/3623954_train_test/checkpoints/best.pt}"

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

# Scenario
: "${SCENARIO:=S3}"

# Exclude Variables
: "${EXCLUDE_VARS:="pre_rolling_mean,tmp_rolling_mean,spfh_rolling_mean,tmax_rolling_mean,tmin_rolling_mean"}"

# ---- Env ----
module purge
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate trendy-gpu
mkdir -p logs

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/a_pipeline/2.predict/predict.py \
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
  --export_nc \
  --exclude_vars "${EXCLUDE_VARS}" \

  #   --overwrite_data \