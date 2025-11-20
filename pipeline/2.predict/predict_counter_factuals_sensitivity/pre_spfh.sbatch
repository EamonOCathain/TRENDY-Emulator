#!/usr/bin/env bash
#SBATCH --job-name=submit_pre_spfh_offsets
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --chdir=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/2.predict/predict_counter_factuals_sensitivity

set -euo pipefail
mkdir -p logs

# Percentages to remove from pre + spfh
PCTS=(0 10 20 30 40 50 60 70 80 90 100)

for pct in "${PCTS[@]}"; do

    # Build "-25%" style string
    offset="-${pct}%"

    echo "Submitting precipitation + humidity offset: ${offset}"

    sbatch --export=ALL,\
FORCING_OFFSETS="daily:pre=${offset},spfh=${offset}",\
JOB_NAME="pre_spfh_minus_${pct}pct" \
        ./predict_generic.sbatch
done