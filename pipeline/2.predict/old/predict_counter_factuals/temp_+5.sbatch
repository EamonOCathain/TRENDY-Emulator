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
sbatch --export=ALL,FORCING_OFFSETS="daily:tmp=+5,tmin=+5,tmax=+5,annual:tmp_rolling_mean=+5,tmax_rolling_mean=+5,tmin_rolling_mean=+5",JOB_NAME='tmp_+5' ./predict_generic.sbatch