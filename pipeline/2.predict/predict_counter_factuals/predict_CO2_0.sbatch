#!/usr/bin/env bash
#SBATCH --job-name=predict
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --chdir=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/2.predict/predict_counter_factuals

set -euo pipefail
mkdir -p logs

# Submit the generic array job with overridden env vars
sbatch --export=ALL,SCENARIO=S0,FORCING_OFFSETS='annual:co2=-100%',JOB_NAME='co2_0' ./predict_generic.sbatch