#!/bin/bash
#SBATCH --job-name=nee
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs

# ---- activate environment ----
module purge
module load gnu12/12.2.0 openmpi4/4.1.4
module load cdo/2.1.1
module load nco/5.1.3

source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate base

# ---- required arguments ----
IN_DIR="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/base_model/no_carry/no_carry_S3/netcdf/S3/full"
OUT_DIR="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/base_model/no_carry/trends_S3"

# ---- run ----
python take_ensmean_nee.py \
    --in-dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S0 \
    --out-dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S0 \
    --scenario S0

python take_ensmean_nee.py \
    --in-dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S1 \
    --out-dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S1 \
    --scenario S1

python take_ensmean_nee.py \
    --in-dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S2 \
    --out-dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S2 \
    --scenario S2

python take_ensmean_nee.py \
    --in-dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S3 \
    --out-dir /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/ensmean_files/S3 \
    --scenario S3

