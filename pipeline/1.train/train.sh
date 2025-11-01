#!/bin/bash
#SBATCH --job-name=Training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=6
#SBATCH --mem=400G
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
  --job_name test_carry_2 \
  --epochs 40 \
  --num_workers 6 \
  --val_frac 0.5 \
  --test_frac 0.1 \
  --early_stop \
  --early_stop_patience 10 \
  --early_stop_min_delta 0 \
  --early_stop_warmup_epochs 0 \
  --block_locs 140 \
  --prefetch_factor 1 \
  --val_prefetch_factor 1 \
  --carry_years 2 \
  --eval_mb_size 1960 \
  --train_mb_size 1960 \
  --model_monthly_mode sequential_months \
  --use_foundation /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/1.train/runs/2025-11-01/3678019_train_carry_1/checkpoints/best.pt \
  --test_only 


# With block_loc 140:
# 0 carry sequential mode = 3920
# 1 carry = 3920/2 = 1960 
# 2 carry = 3920/3 = 1960