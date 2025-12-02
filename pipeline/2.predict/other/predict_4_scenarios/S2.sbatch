#!/usr/bin/env bash
#SBATCH --job-name=predict_S2
#SBATCH --partition=big
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --chdir=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/2.predict/predict_4_scenarios

set -euo pipefail
mkdir -p logs

# Submit the generic job, forcing SCENARIO for this submission.
# --export=ALL ensures the variable reaches the job even if the cluster’s defaults differ.
sbatch --export=ALL,SCENARIO="S2" ./predict_generic.sbatch