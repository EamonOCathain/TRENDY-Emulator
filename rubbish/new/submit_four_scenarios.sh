#!/usr/bin/env bash
set -euo pipefail

# location of this script & siblings
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# load global config (allows env overrides before calling this script)
source "${SOURCE_DIR}/config.sh"

# Optional: dry-run to print sbatch commands without submitting
: "${DRY_RUN:=false}"

submit_predict() {
  local scenario="$1"
  local dep="$2"           # may be empty

  local cmd=(sbatch --parsable
    --job-name "predict_${scenario}"
    --partition "${GPU_PARTITION}"
    --gres "${PREDICT_GRES}"
    --cpus-per-task "${PREDICT_CPUS}"
    --mem "${PREDICT_MEM}"
    --time "${PREDICT_TIME}"
    --array "${PREDICT_ARRAY}"
    --output "${PREDICT_LOG_DIR}/%x_%A_%a.out"
    --error  "${PREDICT_LOG_DIR}/%x_%A_%a.err"
    --export=ALL,SCENARIO="${scenario}"
    "${SOURCE_DIR}/predict.sh"
  )

  if [[ -n "${dep}" ]]; then
    cmd=(sbatch --parsable
      --dependency="afterok:${dep}"
      --job-name "predict_${scenario}"
      --partition "${GPU_PARTITION}"
      --gres "${PREDICT_GRES}"
      --cpus-per-task "${PREDICT_CPUS}"
      --mem "${PREDICT_MEM}"
      --time "${PREDICT_TIME}"
      --array "${PREDICT_ARRAY}"
      --output "${PREDICT_LOG_DIR}/%x_%A_%a.out"
      --error  "${PREDICT_LOG_DIR}/%x_%A_%a.err"
      --export=ALL,SCENARIO="${scenario}"
      "${SOURCE_DIR}/predict.sh"
    )
  fi

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY] ${cmd[*]}"
    echo "0000000"  # fake job id
  else
    "${cmd[@]}"
  fi
}

submit_export() {
  local scenario="$1"
  local predict_job="$2"

  local cmd=(sbatch --parsable
    --job-name "export_${scenario}"
    --partition "${CPU_PARTITION}"
    --cpus-per-task "${EXPORT_CPUS}"
    --mem "${EXPORT_MEM}"
    --time "${EXPORT_TIME}"
    --array "${EXPORT_ARRAY}"
    --output "${EXPORT_LOG_DIR}/%x_%A_%a.out"
    --error  "${EXPORT_LOG_DIR}/%x_%A_%a.err"
    --dependency=afterok:${predict_job}
    --export=ALL,SCENARIO="${scenario}"
    "${SOURCE_DIR}/export_netcdf.sh"
  )

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY] ${cmd[*]}"
    echo "0000000"  # fake job id
  else
    "${cmd[@]}"
  fi
}

# ensure log dirs exist (predict/export scripts also create them, this is just friendly)
mkdir -p "${PREDICT_LOG_DIR}" "${EXPORT_LOG_DIR}"

# chain order: S3 → S2 → S1 → S0
SCENARIOS=(S3 S2 S1 S0)

prev_predict_job=""
declare -A PREDICT_JOBS
declare -A EXPORT_JOBS

for scen in "${SCENARIOS[@]}"; do
  echo "[SUBMIT] Submitting predict for ${scen} (depends on: ${prev_predict_job:-none})"
  pj=$(submit_predict "${scen}" "${prev_predict_job:-}")
  echo "[SUBMIT]   predict_${scen} -> ${pj}"
  PREDICT_JOBS["${scen}"]="${pj}"

  echo "[SUBMIT] Submitting export for ${scen} (depends on its predict ${pj})"
  ej=$(submit_export "${scen}" "${pj}")
  echo "[SUBMIT]   export_${scen} -> ${ej}"
  EXPORT_JOBS["${scen}"]="${ej}"

  # next scenario’s predict waits for THIS scenario’s predict (not export)
  prev_predict_job="${pj}"
done

echo
echo "==== Summary ===="
for scen in "${SCENARIOS[@]}"; do
  printf "%-4s predict=%s  export=%s\n" "${scen}" "${PREDICT_JOBS[$scen]}" "${EXPORT_JOBS[$scen]}"
done