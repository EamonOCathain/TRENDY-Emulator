#!/bin/bash
#SBATCH --job-name=Training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=4


set -eo pipefail
mkdir -p logs
export PYTHONUNBUFFERED=1

# Microbatch size of 2940 is fine without carry in A40 and A100


# threads per GPU worker (adjust to taste)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export SLURM_MEM_BIND=local
export TORCHDYNAMO_DISABLE=1

set +u
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate trendy-gpu
set -u

cd "${SLURM_SUBMIT_DIR}"

RUN_COMMENTS=$(cat <<'EOF'
Updating to step optimiser every batch.
EOF
)

echo "==== GPU inventory ===="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "==== CPU RAM requested ===="
echo "SLURM --mem = 200G (total on node)"
free -h

NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1 

# torchrun sets LOCAL_RANK/RANK/WORLD_SIZE expected by your script
torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} train.py \
  --job_name test_core_branch \
  --subset_frac 0.0001 \
  --epochs 1 \
  --mb_size 1470 \
  --num_workers 8 \
  --val_frac 0.5 \
  --val_freq 0.5 \
  --shuffle_windows \
  --early_stop \
  --early_stop_patience 7 \
  --early_stop_min_delta 0 \
  --early_stop_warmup_epochs 0 \
  --use_foundation /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/1.train/runs/saved_checkpoints/base_model/new_loss/base_model_new_loss/checkpoints/best.pt \
  --transfer_learn \
  --transfer_learn_vars lai_avh15c1 \
  --transfer_learn_years 1982-2018 \

