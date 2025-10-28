#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=ilamb_ensmean
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=6

set -euo pipefail

# ensure log dir exists
mkdir -p logs

# sensible threading for BLAS / numpy
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

# Activate ILAMB env
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate /Net/Groups/BGI/people/ecathain/envs/ilamb

# Paths
cd /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/counter_factuals/test
export ILAMB_ROOT="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/counter_factuals/test"
export MASKS_DIR="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/masks"

# Run ILAMB
/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/analysis/ilamb/ILAMB/bin/ilamb-run \
  --config "$ILAMB_ROOT/build.cfg" \
  --model_root "$ILAMB_ROOT/MODELS/" \
  --define_regions "$MASKS_DIR/ilamb_tvt.nc" \
  --regions global test val train

