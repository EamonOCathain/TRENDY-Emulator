#!/usr/bin/env bash
#SBATCH --job-name=submit_tmp_offsets
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --chdir=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/2.predict/predict_counter_factuals_sensitivity

set -euo pipefail
mkdir -p logs

# 11 offsets from -5 to +5 (inclusive), step 1
OFFSETS=(-5 -4 -3 -2 -1 0 1 2 3 4 5)

for val in "${OFFSETS[@]}"; do

    # Here val is already "-5", "-4", ..., "0", "1", ..., "5"
    offset="${val}"

    # Job name, e.g. tmp_-5, tmp_0, tmp_5
    jobname="tmp_${val}"

    echo "Submitting temperature offset: ${offset}"

    sbatch --export=ALL,\
FORCING_OFFSETS="daily:tmp=${offset},daily:tmin=${offset},daily:tmax=${offset}",\
JOB_NAME="${jobname}" \
        ./predict_generic.sbatch

done