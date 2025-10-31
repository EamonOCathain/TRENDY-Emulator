#!/bin/bash
#SBATCH --job-name=Training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
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
  --job_name test_new_diff \
  --epochs 1 \
  --subset_frac 0.001 \
  --mb_size 2940 \
  --num_workers 4 \
  --val_frac 0.005 \
  --test_frac 0.001 \
  --shuffle_windows \
  --early_stop \
  --early_stop_patience 10 \
  --early_stop_min_delta 0 \
  --early_stop_warmup_epochs 0 \
  --use_foundation /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/checkpoints/base_model/checkpoints/best.pt \
  --block_locs 70 \
  --prefetch_factor 1 \
  --val_prefetch_factor 1 \
  --carry_years 1 \
  --eval_mb_size 1470 
