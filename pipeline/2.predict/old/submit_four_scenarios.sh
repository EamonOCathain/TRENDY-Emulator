#!/usr/bin/env bash
set -euo pipefail

# Usage: ./submit_predict_series.sh path/to/predict.sbatch [--max-parallel N]
# Example:
#   ./submit_predict_series.sh predict.sbatch --max-parallel 1   # strictly sequential
#   ./submit_predict_series.sh predict.sbatch --max-parallel 2   # 2 at a time (S3&S2), then S1 depends on S3, S0 on S2

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sbatch_script> [--max-parallel N]" >&2
  exit 1
fi

SBATCH_SCRIPT=$1
shift || true

MAX_PARALLEL=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-parallel)
      MAX_PARALLEL=${2:-1}
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Error: sbatch script not found: $SBATCH_SCRIPT" >&2
  exit 1
fi

# Submit order: S3 -> S2 -> S1 -> S0
SCENARIOS=(S3 S2 S1 S0)

# Track submitted job IDs to wire dependencies.
declare -a JOB_IDS=()

for i in "${!SCENARIOS[@]}"; do
  scen="${SCENARIOS[$i]}"

  # Set a clearer job name; overrides the #SBATCH --job-name header.
  jobname="predict_${scen}"

  # You already pass --scenario inside the wrapped python, but we export SCENARIO here
  # so your sbatch script can pick it up via `${SCENARIO}`. We also tweak JOB_NAME
  # so your run folder reflects the scenario.
  export_str="ALL,SCENARIO=${scen},JOB_NAME=transfer_learn/avh15c1_lai/no_carry/${scen}"

  # Dependency: afterok on the job that is MAX_PARALLEL positions before this one.
  # E.g. with MAX_PARALLEL=2:
  #   submit S3,S2 (no deps), then S1 depends on S3, S0 depends on S2.
  dep_arg=()
  if (( i >= MAX_PARALLEL )); then
    parent_idx=$(( i - MAX_PARALLEL ))
    parent_jobid="${JOB_IDS[$parent_idx]}"
    dep_arg=(--dependency="afterok:${parent_jobid}")
  fi

  # --parsable returns the numeric jobid (or jobid_array) so we can chain.
  jobid=$(sbatch --parsable \
                 --job-name "$jobname" \
                 --export "$export_str" \
                 "${dep_arg[@]}" \
                 "$SBATCH_SCRIPT")

  echo "Submitted $scen as job $jobid ${dep_arg:+(dep ${dep_arg[*]})}"
  JOB_IDS+=("$jobid")
done

echo "All submissions dispatched."