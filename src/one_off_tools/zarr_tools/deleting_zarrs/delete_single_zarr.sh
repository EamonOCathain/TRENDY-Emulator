#!/bin/bash
#SBATCH --job-name=delete_zarr
#SBATCH --output=logs/delete_zarr_%j.out
#SBATCH --error=logs/delete_zarr_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Usage:
#   sbatch delete_zarr.sh /path/to/store.zarr

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/target.zarr"
    exit 1
fi

TARGET=$1

# Safety checks
if [[ ! -d "$TARGET" ]]; then
    echo "[ERROR] $TARGET is not a directory."
    exit 2
fi
if [[ "${TARGET##*.}" != "zarr" ]]; then
    echo "[ERROR] Refusing to delete: $TARGET does not end with .zarr"
    exit 3
fi

echo "[INFO] Deleting Zarr store: $TARGET"
rm -rf -- "$TARGET"
echo "[INFO] Deleted: $TARGET"