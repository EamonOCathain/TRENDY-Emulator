#!/usr/bin/env bash
# wrapper.sh
# Submit mass-balance (array) job, then submit merge job with dependency.

set -euo pipefail

# Paths to your job scripts (edit if you keep them elsewhere)
MB_JOB_SCRIPT="mass_balance_labels.sh"
MERGE_JOB_SCRIPT="merge_shards_mass_balance.sh"

# Submit the shard job
SUBMIT_OUT=$(sbatch "$MB_JOB_SCRIPT")
echo "$SUBMIT_OUT"

# Extract the job ID (works with output like: "Submitted batch job 3542467")
MB_JOB_ID=$(awk '{print $4}' <<<"$SUBMIT_OUT")

if [[ -z "${MB_JOB_ID:-}" ]]; then
  echo "Failed to parse job ID from sbatch output:"
  echo "$SUBMIT_OUT"
  exit 1
fi

echo "Mass-balance array submitted as JobID: ${MB_JOB_ID}"

# Submit the merge job with dependency on successful completion of the array
sbatch --dependency=afterok:${MB_JOB_ID} "$MERGE_JOB_SCRIPT"