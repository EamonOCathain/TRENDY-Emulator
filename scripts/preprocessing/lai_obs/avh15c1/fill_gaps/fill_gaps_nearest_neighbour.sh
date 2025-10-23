#!/bin/bash
#SBATCH --job-name=lai_gapfill
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
# (Only needed if your site requires modules; otherwise skip)
# module load python/3.10  # example

source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

python -u fill_gaps_nearest_neighbour.py \
  --in /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/lai_avh15c1.nc \
  --out /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/lai_avh15c1_gap_filled.nc \
  --mask /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks/tvt_mask.nc \
  --var lai_avh15c1 \
  --year-min 1982 --year-max 2018 \
  --out-chunks time:-1,lat:1,lon:1 \
  --workers 8 \
  --compressor zstd --clevel 1