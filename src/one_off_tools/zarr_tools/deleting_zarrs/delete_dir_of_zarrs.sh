#!/bin/bash
#SBATCH --job-name=delete_zarrs_rr
#SBATCH --output=logs/delete_zarrs_rr_%A_%a.out
#SBATCH --error=logs/delete_zarrs_rr_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-9
# Submit like:
#   sbatch delete_zarrs_rr.sh /path/to/root_dir
# (10-way round robin in this example)

set -euo pipefail

# ---------- args ----------
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/root_dir" >&2
  exit 1
fi
ROOT="$1"

if [[ ! -d "$ROOT" ]]; then
  echo "[ERROR] '$ROOT' is not a directory" >&2
  exit 2
fi

# ---------- slurm info ----------
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TASK_CNT=${SLURM_ARRAY_TASK_COUNT:-1}
JOB_ID=${SLURM_JOB_ID:-$$}

# Where to place the manifest so all array elements see the same list
MANIFEST_DIR=${MANIFEST_DIR:-logs}
mkdir -p "$MANIFEST_DIR"
MANIFEST="${MANIFEST_DIR}/zarr_delete_${JOB_ID}.list"
READY_FLAG="${MANIFEST}.ready"

echo "[INFO] Job $JOB_ID array $TASK_ID/$((TASK_CNT-1)) scanning root: $ROOT"
echo "[INFO] Manifest: $MANIFEST"

# ---------- create the manifest once (task 0), others wait ----------
if [[ "$TASK_ID" == "0" ]]; then
  echo "[INFO] Task 0 building manifest..."
  # Use find -print0 + sort -z to be robust to spaces/newlines in paths.
  # Then write one path per line to MANIFEST.
  tmp="${MANIFEST}.tmp"

  # shellcheck disable=SC2016
  find "$ROOT" -type d -name '*.zarr' -print0 \
    | sort -z \
    | tr '\0' '\n' > "$tmp"

  COUNT_TOTAL=$(wc -l < "$tmp" | awk '{print $1}')
  echo "[INFO] Found $COUNT_TOTAL .zarr store(s) under $ROOT"

  mv "$tmp" "$MANIFEST"
  echo "$COUNT_TOTAL" > "$READY_FLAG"
else
  # Wait up to 60 minutes for manifest to appear
  MAX_WAIT_SEC=${MAX_WAIT_SEC:-3600}
  SLEEP=5
  elapsed=0
  echo "[INFO] Waiting for manifest up to $MAX_WAIT_SEC s..."
  while (( elapsed < MAX_WAIT_SEC )); do
    if [[ -f "$READY_FLAG" && -f "$MANIFEST" ]]; then
      break
    fi
    sleep "$SLEEP"
    elapsed=$((elapsed + SLEEP))
  done
  if [[ ! -f "$MANIFEST" ]]; then
    echo "[ERROR] Timed out waiting for manifest $MANIFEST after $elapsed s" >&2
    exit 3
  fi
fi

COUNT_TOTAL=$(cat "$READY_FLAG" 2>/dev/null || wc -l < "$MANIFEST" | awk '{print $1}')
echo "[INFO] Total .zarr to delete: ${COUNT_TOTAL:-unknown}"

# ---------- load manifest and compute my round-robin slice ----------
# Read safely even with spaces; paths are one per line.
mapfile -t ZARRS < "$MANIFEST"
N=${#ZARRS[@]}

if (( N == 0 )); then
  echo "[INFO] No .zarr stores to delete. Exiting."
  exit 0
fi

# Build my indices: i such that i % TASK_CNT == TASK_ID
MY_INDEXES=()
for ((i=TASK_ID; i<N; i+=TASK_CNT)); do
  MY_INDEXES+=("$i")
done

MY_COUNT=${#MY_INDEXES[@]}
echo "[INFO] This task will delete $MY_COUNT of $N stores (round-robin)."

# Show a short preview
for ((k=0; k<MY_COUNT && k<10; k++)); do
  echo "  - ${ZARRS[${MY_INDEXES[$k]}]}"
done
(( MY_COUNT > 10 )) && echo "  ... and $((MY_COUNT-10)) more"

# ---------- delete loop ----------
DELETED=0
SKIPPED=0
FAILED=0

for idx in "${MY_INDEXES[@]}"; do
  target="${ZARRS[$idx]}"

  # Safety checks
  if [[ -z "$target" ]]; then
    ((SKIPPED++))
    echo "[SKIP] empty path at index $idx"
    continue
  fi
  if [[ ! -d "$target" ]]; then
    ((SKIPPED++))
    echo "[SKIP] not a directory: $target"
    continue
  fi
  if [[ "${target##*.}" != "zarr" ]]; then
    ((SKIPPED++))
    echo "[SKIP] not *.zarr: $target"
    continue
  fi

  echo "[INFO] Deleting ($((DELETED+SKIPPED+FAILED+1))/${MY_COUNT}) -> $target"
  # Use rm -rf on the resolved path to avoid oddities with relative paths
  if rm -rf -- "$target"; then
    ((DELETED++))
    echo "[OK] Deleted: $target"
  else
    ((FAILED++))
    echo "[ERROR] Failed to delete: $target"
  fi
done

echo "[SUMMARY] Task $TASK_ID: deleted=$DELETED skipped=$SKIPPED failed=$FAILED (of $MY_COUNT assigned; total=$N)"