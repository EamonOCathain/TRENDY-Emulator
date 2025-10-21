#!/bin/bash
#SBATCH --job-name=plot_maps
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-59

# ---------- User-configurable env vars (can override at submit time) ----------
: "${LABELS_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference}"
: "${PREDS_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/first_full_run}"
: "${OUT_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/analysis/timestep_maps/plots/timesteps}"
: "${DATA_CMAP:=viridis}"
: "${BIAS_CMAP:=RdBu_r}"
: "${OVERWRITE:=true}"

# ---------- Environment ----------
module purge
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base   # CPU env is enough
mkdir -p logs

# ---------- Build args ----------
ARGS=( 
  --labels_dir "$LABELS_DIR" 
  --preds_dir "$PREDS_DIR" 
  --out_dir "$OUT_DIR" 
)

if [[ -n "$DATA_CMAP" ]]; then
  ARGS+=( --data_cmap "$DATA_CMAP" )
fi
if [[ -n "$BIAS_CMAP" ]]; then
  ARGS+=( --bias_cmap "$BIAS_CMAP" )
fi
case "${OVERWRITE,,}" in
  1|true|yes) ARGS+=( --overwrite ) ;;
esac

# ---------- Run ----------
echo "[INFO] Starting task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
python -u first_last_timesteps.py "${ARGS[@]}"