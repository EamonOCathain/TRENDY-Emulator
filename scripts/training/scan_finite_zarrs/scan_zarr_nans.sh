#!/bin/bash
#SBATCH --mem=8G
#SBATCH --array=0-63 
#SBATCH --job-name=scan_finite
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

ROOT="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/training_new"
OUTDIR="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/training/other/scan_finite_zarrs/out"
mkdir -p "$OUTDIR" logs

# One CSV per task; you can merge later
python scan_zarr_nans.py --root "$ROOT" --include-glob "*.zarr" \
  --arrays x m a \
  --tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID \
  --block-size 4096 \
  --csv "$OUTDIR/findings_${SLURM_ARRAY_TASK_ID}.csv" \
  --summary-json "$OUTDIR/summary_${SLURM_ARRAY_TASK_ID}.json"