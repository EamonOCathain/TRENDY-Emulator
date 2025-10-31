#!/bin/bash
#SBATCH --job-name=mb_sweep_carry_123
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs
export PYTHONUNBUFFERED=1

# threads per GPU worker (adjust to taste)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export SLURM_MEM_BIND=local
export TORCHDYNAMO_DISABLE=1

# make DDP fail fast if a rank OOMs
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# optional per-trial timeout
TIMEOUT_PER_TRIAL=20m

# env
set +u
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate trendy-gpu
set -u

cd "${SLURM_SUBMIT_DIR}"

TRAIN_SCRIPT="/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/1.train/train.py"
NPROC="${SLURM_GPUS_ON_NODE:-1}"

# --------- function to try a given train_mb_size ----------
try_run () {
  local MB="$1"
  echo "==> Trying train_mb_size=${MB}"
  if /usr/bin/timeout "${TIMEOUT_PER_TRIAL}" \
     torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
        --job_name "sweep_mb_${MB}" \
        --epochs 1 \
        --subset_frac 0.001 \
        --num_workers 8 \
        --val_frac 1 \
        --test_frac 1 \
        --scheduler none \
        --lr 9e-5 \
        --loss_type mse \
        --train_only \
        --carry_years 123 \
        --model_monthly_mode sequential_months \
        --train_mb_size "${MB}" \
        --eval_mb_size 1470 \
        --block_locs 70 \
        --prefetch_factor 1 \
        --train_only \
        --val_prefetch_factor 1; then
    echo "   ✔ train_mb_size=${MB} OK"
    return 0
  else
    echo "   ✘ train_mb_size=${MB} FAILED"
    return 1
  fi
}

# --------- exponential growth until first failure ----------
LOW=1
HIGH=1470
echo "Starting exponential sweep…"
while true; do
  if try_run "${HIGH}"; then
    LOW="${HIGH}"
    if [ "${HIGH}" -lt 4096 ]; then
      HIGH=$((HIGH*2))
    else
      HIGH=$((HIGH+512))
    fi
  else
    echo "First failure at train_mb_size=${HIGH}"
    break
  fi
  if [ "${HIGH}" -gt 8192 ]; then
    echo "Reached safety cap without failing; using ${LOW}"
    echo "Max successful train_mb_size=${LOW}"
    exit 0
  fi
done

# --------- binary search between LOW (ok) and HIGH (fail) ----------
LEFT="${LOW}"
RIGHT=$((HIGH-1))
BEST="${LEFT}"

echo "Binary search between ${LEFT}..${RIGHT}"
while [ "${LEFT}" -le "${RIGHT}" ]; do
  MID=$(( (LEFT + RIGHT) / 2 ))
  if try_run "${MID}"; then
    BEST="${MID}"
    LEFT=$((MID+1))
  else
    RIGHT=$((MID-1))
  fi
done

echo "======================================"
echo " Max successful train_mb_size = ${BEST}"
echo " First failing train_mb_size  = ${HIGH}"
echo "======================================"