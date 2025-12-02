#!/bin/bash
#SBATCH --job-name=analyse_metrics
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-63
#SBATCH --chdir=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/4.Analyse/create_csvs/iav

set -euo pipefail
mkdir -p logs

# ---- Env ----
module purge
set +u
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base
set -u

# ---- Run: one scenario per array task ----
python -u iav.py --mask_test_only