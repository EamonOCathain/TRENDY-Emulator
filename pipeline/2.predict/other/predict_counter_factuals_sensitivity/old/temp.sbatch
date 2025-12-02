#!/usr/bin/env bash
#SBATCH --job-name=submit_temp_offsets
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --chdir=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/2.predict/predict_counter_factuals_sensitivity

set -euo pipefail
mkdir -p logs

# Temperature offsets to run (°C)
OFFSETS=(-8 -6 -4 -2 0 +2 +4 +6 +8)

for off in "${OFFSETS[@]}"; do

    # Build signed string for ILAMB input (e.g., +2 or -6)
    if (( off > 0 )); then
        sign="+${off}"
    else
        sign="${off}"
    fi

    echo "Submitting temperature offset experiment: ${sign}"

    sbatch --export=ALL,\
FORCING_OFFSETS="daily:tmp=${sign},tmin=${sign},tmax=${sign},annual:tmp_rolling_mean=${sign},tmin_rolling_mean=${sign},tmax_rolling_mean=${sign}",\
JOB_NAME="tmp_${sign}" \
        ./predict_generic.sbatch
done