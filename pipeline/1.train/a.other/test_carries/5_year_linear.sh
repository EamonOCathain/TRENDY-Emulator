#!/bin/bash
#SBATCH --job-name=Testing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
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
torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/1.train/train.py \
  --job_name test_5_year_linear \
  --epochs 40 \
  --num_workers 4 \
  --val_frac 0.5 \
  --test_frac 0.1 \
  --early_stop \
  --early_stop_patience 10 \
  --early_stop_min_delta 0 \
  --early_stop_warmup_epochs 0 \
  --block_locs 140 \
  --prefetch_factor 1 \
  --val_prefetch_factor 1 \
  --carry_years 4 \
  --eval_mb_size 1880 \
  --train_mb_size 1880 \
  --model_monthly_mode sequential_months \
  --use_foundation /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/1.train/runs/2025-11-06/3719656_train_carry_5/checkpoints/best.pt \
  --test_only \

# With block_loc 140 total windows without carry is 11760..
# 0 carry sequential mode = 11760 total windows = 84 windows per location = 3920
# 1 carry = 11620 total windows = 83 per location = 3920/2 = 1960 
# 2 carry = 11480 total windows = 82 per location = 3920/3 = 1960 (but eval less - 980 works, trying 1470)
# 4 carry = 11200 total windows = 80 per location = 1960 train and val.. 
# 8 carry = 10640 total windows = 76 per location = 
# 16 carry = 9520 total windows = 68 per location =  

# 0 carry = 11900 total windows = 85 per location = 3920
# 1 carry = 11760 total windows = 84 per location = 2940
# 2 carry = 11620 total windows = 83 per location = 1960 train, 1470 val
# 4 carry = 11340 total windows = 81 per location = 1880
# 8 carry = 10780 total windows = 77 per location =
# 16 carry = 9680 total windows = 69 per location =
# 32 carry = 7420 total windows = 53 per location =
# 64 carry = 2940 total windows = 21 per location =
# 84 carry = 140 total windows = 1 per location =