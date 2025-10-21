#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=mk_tiles
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-90%91

set -euo pipefail
mkdir -p logs

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLOSC_NTHREADS=1
export VAR_WORKERS=1

source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# So this script now arrays first by the tensors to be created (time-res and location/period combos)
# Then it sub arrays into the location tiles for that tensor (each being 70 locations)

# To initialise the zarrs dont unset INIT_ONLY, instead set it to 1. But only needs to be done once.
# export INIT_ONLY=1
# python -u make_training_tiles.py --daily_files_mode twenty

# To figure out the number of tiles in task 6:
# INIT_ONLY=1 SELECT_TASK_IDX=6 python -u make_training_tiles.py --daily_files_mode twenty

# To run the tiles in task 6: 
# sbatch --export=ALL,SELECT_TASK_IDX=6 make_training_tiles.sh

# To finalise after all arrays are done:
# export FINALIZE=1
# export SELECT_TASK_IDX=6
# python -u make_training_tiles.py --daily_files_mode twenty --validate

# To run a single task, and specific arrays:
# sbatch --array=79-90 --export=ALL,SELECT_TASK_IDX=6,SHARD_COUNT=91 make_training_tiles.sh

# SELECT_TASK_IDX is passed via --export above
unset INIT_ONLY
python -u make_training_tiles.py --daily_files_mode twenty