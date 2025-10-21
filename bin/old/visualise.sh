#!/bin/bash
#SBATCH --job-name=visualise
#SBATCH --partition=big
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-14

set -euo pipefail

# ---------- User-configurable (override at submit with VAR=...) ----------
: "${LABELS_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference}"
: "${PREDS_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/first_full_run}"
: "${OUTPUT_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/plots/first_full_run}"

# ---------- Environment ----------
module purge
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base  

# ---------- Build args ----------
ARGS=(
  --labels_dir "$LABELS_DIR"
  --preds_dir  "$PREDS_DIR"
  --output_dir "$OUTPUT_DIR"
)

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
# Optional: keep dask’s work scratch on local fast storage
export DASK_TEMPORARY_DIRECTORY=${TMPDIR:-/tmp}

# ---------- Run ----------
echo "[INFO] Task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
python -u visualise_old.py "${ARGS[@]}"