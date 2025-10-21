#!/bin/bash
#SBATCH --job-name=infer_tiles
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-7
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err


set -euo pipefail

# -------- Defaults (override by exporting before running) --------
JOB_NAME="${JOB_NAME:-nudge_lambda_0.01}"
FORCING_DIR="${FORCING_DIR:-/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference}"
WEIGHTS="${WEIGHTS:-/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/training/runs/first_full_run/3524889_full_run/checkpoints/epoch43.pt}"
STORE_PERIOD="${STORE_PERIOD:-1901-01-01:2023-12-31}"
WRITE_PERIOD="${WRITE_PERIOD:-1901-01-01:2023-12-31}"
NUDGE_LAMBDA="${NUDGE_LAMBDA:-0.01}"

# Array size & resources to pass to infer_tiles job
SHARDS="${SHARDS:-8}"                   # array will be 0..SHARDS-1
PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-1}"
MEM="${MEM:-24G}"
CPUS="${CPUS:-4}"
TIME="${TIME:-24:00:00}"

# Path to your existing sbatch script
SBATCH_SCRIPT="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/inference/infer_tiles.sh"

# -------- Helper to submit one scenario with optional dependency --------
submit_scn () {
  local SCN="$1"
  local DEP_OPT=()
  if [[ $# -ge 2 && -n "${2:-}" ]]; then
    DEP_OPT=(--dependency="afterany:${2}")
  fi

  sbatch --parsable \
    -p "${PARTITION}" \
    --gres="gpu:${GPUS}" \
    --mem="${MEM}" \
    --cpus-per-task="${CPUS}" \
    --time="${TIME}" \
    --array="0-$((SHARDS-1))" \
    "${DEP_OPT[@]}" \
    --export=ALL,JOB_NAME="${JOB_NAME}",SCENARIO="${SCN}",FORCING_DIR="${FORCING_DIR}",WEIGHTS="${WEIGHTS}",STORE_PERIOD="${STORE_PERIOD}",WRITE_PERIOD="${WRITE_PERIOD}",NUDGE_LAMBDA="${NUDGE_LAMBDA}" \
    "${SBATCH_SCRIPT}"
}

echo "[INFO] Submitting chained scenarios with ${SHARDS} shards each"

jid0=$(submit_scn S0)
echo "[INFO] S0 array submitted: ${jid0}"

jid1=$(submit_scn S1 "${jid0}")
echo "[INFO] S1 array submitted (after:${jid0}): ${jid1}"

jid2=$(submit_scn S2 "${jid1}")
echo "[INFO] S2 array submitted (after:${jid1}): ${jid2}"

jid3=$(submit_scn S3 "${jid2}")
echo "[INFO] S3 array submitted (after:${jid2}): ${jid3}"

echo "[DONE] Chain: S0(${jid0}) -> S1(${jid1}) -> S2(${jid2}) -> S3(${jid3})"