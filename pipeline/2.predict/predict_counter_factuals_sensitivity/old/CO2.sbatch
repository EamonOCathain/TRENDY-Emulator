#!/usr/bin/env bash
#SBATCH --job-name=co2_sensitivity
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --chdir=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/2.predict/predict_counter_factuals_sensitivity

set -euo pipefail
mkdir -p logs

CO2_LEVELS=(75 150 225 300 375 450 525 600)

for level in "${CO2_LEVELS[@]}"; do
    echo "Submitting CO₂ = ${level} ppm"

    sbatch --export=ALL,SCENARIO=S0,\
FORCING_OFFSETS="annual:co2=${level}",\
JOB_NAME="co2_${level}ppm" \
        ./predict_generic.sbatch
done