#!/bin/bash
#SBATCH --job-name=infer_tiles
#SBATCH --partition=gpu
#SBATCH --ntasks=8
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-25%10
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=gpu:1

set -euo pipefail

# ---- USER PARAMS ----
: "${JOB_NAME:=z_adaptive}"
: "${SCENARIO:=S3}"
: "${FORCING_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference}"
: "${OUT_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/nudge_array_check/z_adaptive}"
: "${WEIGHTS:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/training/runs/full_run_with_mrso/checkpoints/best.pt}"
: "${STORE_PERIOD:=1901-01-01:2023-12-31}"
: "${WRITE_PERIOD:=1901-01-01:2023-12-31}"
: "${NUDGE_MODE:=z_adaptive}"
: "${CARRY_FORWARD_STATES:=True}"
# sweep params
MIN=0.25
MAX=0.50
STEP=0.01
DECIMALS=2  

# build sweep: include 0 explicitly + the MIN..MAX range, then dedupe
ZERO="$(printf "%.${DECIMALS}f" 0)"
RANGE_VALUES=($(seq -f "%.${DECIMALS}f" "$MIN" "$STEP" "$MAX"))
declare -A seen=()
NUDGE_VALUES=()
for v in "$ZERO" "${RANGE_VALUES[@]}"; do
  if [[ -z "${seen[$v]:-}" ]]; then
    NUDGE_VALUES+=("$v")
    seen[$v]=1
  fi
done

echo "[INFO] NUDGE_VALUES: ${NUDGE_VALUES[@]}"
echo "[INFO] Total arrays: ${#NUDGE_VALUES[@]}"
echo "[HINT] For this sweep, set: #SBATCH --array=0-$((${#NUDGE_VALUES[@]}-1))%5"

# validate SLURM_ARRAY_TASK_ID
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || ${SLURM_ARRAY_TASK_ID} -lt 0 || ${SLURM_ARRAY_TASK_ID} -ge ${#NUDGE_VALUES[@]} ]]; then
  echo "Bad SLURM_ARRAY_TASK_ID='${SLURM_ARRAY_TASK_ID}' (expected 0..$((${#NUDGE_VALUES[@]}-1)))"
  exit 1
fi

# assign this task’s lambda
NUDGE_LAMBDA="${NUDGE_VALUES[SLURM_ARRAY_TASK_ID]}"
ARRAY_NAME="nudge_${NUDGE_LAMBDA}"

echo "[INFO] Using NUDGE_LAMBDA=${NUDGE_LAMBDA}, ARRAY_NAME=${ARRAY_NAME}"
mkdir -p logs

# ---- Env ----
module purge
set +u                    
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate trendy-gpu
set -u  

echo "[INFO] Running with NUDGE_LAMBDA=${NUDGE_LAMBDA} for array task ${SLURM_ARRAY_TASK_ID}"

python -u /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/inference/predict.py \
  --tile_index 217 \
  --job_name "${JOB_NAME}" \
  --out_dir "${OUT_DIR}" \
  --scenario "${SCENARIO}" \
  --forcing_dir "${FORCING_DIR}" \
  --weights "${WEIGHTS}" \
  --store_period "${STORE_PERIOD}" \
  --write_period "${WRITE_PERIOD}" \
  --device cuda \
  --nudge_lambda "${NUDGE_LAMBDA}" \
  --nudge_mode "${NUDGE_MODE}" \
  --array_name "${ARRAY_NAME}" \
  --carry_forward_states "${CARRY_FORWARD_STATES:=True}"