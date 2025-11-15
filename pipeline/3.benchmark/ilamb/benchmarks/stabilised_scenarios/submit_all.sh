#!/usr/bin/env bash
# Recursively traverse all subdirectories and run `sbatch submit.sh` in each.

set -euo pipefail

ROOT="$(pwd)"
echo "[INFO] Launching recursive sbatch submissions from: $ROOT"
echo

# Find all submit.sh files recursively
mapfile -t scripts < <(find "$ROOT" -type f -name "submit.sh" | sort)

if [[ ${#scripts[@]} -eq 0 ]]; then
  echo "[WARN] No submit.sh files found under $ROOT"
  exit 0
fi

for script in "${scripts[@]}"; do
  dir="$(dirname "$script")"
  echo "[SUBMIT] $dir"
  (cd "$dir" && sbatch submit.sh)
done

echo
echo "[DONE] Submitted ${#scripts[@]} jobs."