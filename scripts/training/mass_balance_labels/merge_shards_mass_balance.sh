#!/bin/bash
#SBATCH --job-name=mb_merge
#SBATCH --partition=work
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

module purge
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

python -u /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/training/other/mass_balance_labels/merge_shards_mass_balance.py \
  --in_dir  /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/training/other/mass_balance_labels/shards \
  --out_dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/training/other/mass_balance_labels/plots \
  --splits train