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

# Offsets: 0, ±20, ±40, ±60, ±80, ±100, +120, +140, +160, +180, +200
OFFSETS=(0 -20 -40 -60 -80 -100 20 40 60 80 100 120 140 160 180 200)

for pct in "${OFFSETS[@]}"; do

    # Build signed percentage string: "-20%" or "+40%" or "0%"
    if [[ $pct -ge 0 ]]; then
        offset="+${pct}%"
    else
        offset="${pct}%"
    fi

    # Job name: tmp_-20pct, tmp_40pct, tmp_120pct, etc
    jobname="tmp_${pct}pct"

    echo "Submitting temperature offset: ${offset}"

    sbatch --export=ALL,\
FORCING_OFFSETS="daily:tmp=${offset},tmin=${offset},tmax=${offset}",\
JOB_NAME="${jobname}" \
        ./predict_generic.sbatch

done