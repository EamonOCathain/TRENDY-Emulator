#!/bin/bash
#SBATCH --job-name=lai
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4

set -euo pipefail

# Ensure log dir exists
mkdir -p logs

# Modules
module purge
module load gnu12/12.2.0 openmpi4/4.1.4
module load cdo/2.1.1
module load nco/5.1.3

# Optional: reduce thread oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# Conda env
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

IN=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/a_pipeline/3.benchmark/ilamb/ground_truth/global/DATA/lai/AVH15C1/lai.nc
OUT=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/preprocessed/transfer_learning/avh15c1/avh15c1_lai.nc
mkdir -p "$(dirname "$OUT")"

# Run (ensure the script name matches your file)
srun -c "$SLURM_CPUS_PER_TASK" python -u preprocess_avh15c1.py "$IN" "$OUT"