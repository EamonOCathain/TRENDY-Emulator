#!/bin/bash
#SBATCH --job-name=analyse_metrics
#SBATCH --partition=big
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs

# ---- Env ----
module purge
set +u
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
# Use an env that has numpy/xarray/dask/matplotlib
conda activate base
set -u

# ---- Paths ----
JOB_NAME="base_model/carry_no_nudge_sequential_months"
PREDS_DIR="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/base_model_new_loss/carry_no_nudge_sequential_months"
LABELS_DIR="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference"
PLOT_DIR="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/plots"
MASK_NC="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/tvt_mask.nc"

# ---- Panel mapping (index -> panel key) ----
PANELS=(combined test_locs_full global_early global_late)
PANEL="${PANELS[$SLURM_ARRAY_TASK_ID]}"

# Optional knobs (match your previous run style)
NCOLS=3
MAX_POINTS=1000

echo "[INFO] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> panel=$PANEL"

# ---- Run: one figure per array task ----
python -u r2.py \
  --job_name "${JOB_NAME}" \
  --preds_dir "${PREDS_DIR}" \
  --labels_dir "${LABELS_DIR}" \
  --plot_dir "${PLOT_DIR}" \
  --mask_nc "${MASK_NC}" \
  --ncols ${NCOLS} \
  --max_points ${MAX_POINTS} \
  --panel "${PANEL}"