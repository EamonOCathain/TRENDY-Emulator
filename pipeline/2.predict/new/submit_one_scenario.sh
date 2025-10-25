#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SOURCE_DIR}/config.sh"

# Submit GPU prediction array
predict_jobid=$(sbatch --parsable \
  --job-name "predict_${SCENARIO}" \
  --partition "${GPU_PARTITION}" \
  --gres "${PREDICT_GRES}" \
  --cpus-per-task "${PREDICT_CPUS}" \
  --mem "${PREDICT_MEM}" \
  --time "${PREDICT_TIME}" \
  --array "${PREDICT_ARRAY}" \
  --output "${PREDICT_LOG_DIR}/%x_%A_%a.out" \
  --error  "${PREDICT_LOG_DIR}/%x_%A_%a.err" \
  --export=ALL \
  "${SOURCE_DIR}/predict.sh")

echo "[SUBMIT] predict array submitted: ${predict_jobid}"

# Submit CPU export array (depends on predict success)
export_jobid=$(sbatch --parsable \
  --job-name "export_${SCENARIO}" \
  --partition "${CPU_PARTITION}" \
  --cpus-per-task "${EXPORT_CPUS}" \
  --mem "${EXPORT_MEM}" \
  --time "${EXPORT_TIME}" \
  --array "${EXPORT_ARRAY}" \
  --output "${EXPORT_LOG_DIR}/%x_%A_%a.out" \
  --error  "${EXPORT_LOG_DIR}/%x_%A_%a.err" \
  --dependency "afterok:${predict_jobid}" \
  --export=ALL \
  "${SOURCE_DIR}/export_netcdf.sh")

echo "[SUBMIT] export array submitted with dependency on ${predict_jobid}: ${export_jobid}"