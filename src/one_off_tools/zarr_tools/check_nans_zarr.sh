#!/bin/bash
#SBATCH --job-name=check_nans_rr
#SBATCH --output=logs/check_nans_rr_%A_%a.out
#SBATCH --error=logs/check_nans_rr_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-21
#SBATCH --partition=work

module load python  # if needed
srun python scan_zarr_nans.py /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/training_rechunked_for_carry_70 --out out/nans_${SLURM_ARRAY_TASK_ID}.csv