#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --cpus-per-task=8
#SBATCH --partition=big
#SBATCH --mem=35G
#SBATCH --array=0-29
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# #SBATCH --gres=gpu:1

set -euo pipefail

# ---- USER PARAMS ----
: "${JOB_NAME:=TL_32_year_1982_2018}"
: "${CARRY_FORWARD_STATES:=True}"
: "${SEQUENTIAL_MONTHS:=True}"
: "${SCENARIO:=S3}"

# Device
: "${DEVICE:=cpu}"
: "${ILAMB_DIR_GLOBAL:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/transfer_learning/MODELS}"
: "${NUMBER_TILES:=6}"

# Weights
: "${WEIGHTS:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/checkpoints/transfer_learning/32_year/checkpoints/best.pt}"

# Paths
: "${FORCING_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference}"
: "${OUT_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/transfer_learn}"

# Periods
: "${STORE_PERIOD:=1982-01-01:2018-12-31}"
: "${WRITE_PERIOD:=1982-01-01:2018-12-31}"

# TL Vars (predict.py requires --tl_initial_state when --tl_vars is provided)
: "${TL_VARS:=lai_avh15c1}"
: "${TL_INITIAL_STATE:=1982}"

# ---- Env ----
module purge

set -euo pipefail

# Define MKL vars so the conda activate.d hook won't hit nounset
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=INTEL

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
  --carry_forward_states "${CARRY_FORWARD_STATES}" \
  --sequential_months "${SEQUENTIAL_MONTHS}" \
  --device "${DEVICE}" \
  --ilamb_dir_global "${ILAMB_DIR_GLOBAL}" \
  --number_tiles "${NUMBER_TILES}" \
  --export_nc \
  --tl_initial_state "${TL_INITIAL_STATE}" \
  --tl_vars ${TL_VARS} \
  # add --export_nc if you want NetCDFs written after Zarr export