#!/bin/bash
#SBATCH --job-name=mb_sweep_sequential
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=6
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=4

set -euo pipefail
mkdir -p logs
export PYTHONUNBUFFERED=1

# Threads per GPU worker (tune if dataloader stalls)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export SLURM_MEM_BIND=local
export TORCHDYNAMO_DISABLE=1

# Optional: safer NCCL behavior for multi-GPU runs
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Debugging helpers
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

# Per-trial timeout (prevents a single bad MB from stalling the sweep)
TIMEOUT_PER_TRIAL=20m

# ---- Env ----
set +u
source /User/homes/ecathain/miniconda3/etc/profile.d/conda.sh
conda activate trendy-gpu
set -u

cd "${SLURM_SUBMIT_DIR}"

echo "==== GPU inventory ===="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
echo "==== CPU/RAM requested ===="
echo "SLURM --mem = ${SLURM_MEM_PER_NODE:-400G} (total on node)"
free -h || true
echo "======================================"

# torchrun: number of processes = number of GPUs on the node
NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l | tr -d ' ')}

# Training entrypoint (use your repo path if not in CWD)
TRAIN_SCRIPT="train.py"

# Common args for your sequential monthly mode with 1-year carry
COMMON_ARGS=(
  --job_name sweep_sequential_mb
  --subset_frac 0.10              # keep this relatively high to stress memory realistically
  --epochs 1                      # 1 epoch per probe is enough
  --num_workers 6
  --val_frac 0.5
  --test_frac 0.2
  --early_stop
  --early_stop_patience 10
  --early_stop_min_delta 0
  --early_stop_warmup_epochs 0
  --block_locs 140
  --prefetch_factor 1
  --val_prefetch_factor 1
  --carry_years 1
  --model_monthly_mode sequential_months
  --eval_mb_size 1960
  --scheduler none
  --lr 9e-5
  --loss_type mse
  --use_foundation /Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/checkpoints/base_model/checkpoints/best.pt
  --train_only                   # sweep focuses on fitting stability/speed
)

# ---------- function to try a given train_mb_size ----------
try_run () {
  local MB="$1"
  echo "==> Trying train_mb_size=${MB}"
  if /usr/bin/timeout "${TIMEOUT_PER_TRIAL}" \
     torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
        "${COMMON_ARGS[@]}" \
        --train_mb_size "${MB}"; then
    echo "   ✔ train_mb_size=${MB} OK"
    return 0
  else
    echo "   ✘ train_mb_size=${MB} FAILED"
    return 1
  fi
}

# ---------- Exponential sweep to first failure ----------
LOW=1960     # start from a known-good baseline for A100 (adjust if you like)
HIGH=2940    # common target when not carrying; we’re in carry=1, so verify
echo "Starting exponential sweep…"
while true; do
  if try_run "${HIGH}"; then
    LOW="${HIGH}"
    # Grow conservatively after ~3k to avoid long OOM retries
    if [ "${HIGH}" -lt 4096 ]; then
      HIGH=$((HIGH*2))
    else
      HIGH=$((HIGH+512))
    fi
  else
    echo "First failure at train_mb_size=${HIGH}"
    break
  fi
  if [ "${HIGH}" -gt 16384 ]; then
    echo "Reached safety cap without failing; using ${LOW}"
    echo "Max successful train_mb_size=${LOW}"
    exit 0
  fi
done

# ---------- Binary search between LOW(ok) and HIGH(fail) ----------
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

# Optional: kick off a real multi-epoch training run at the found MB
# torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
#   "${COMMON_ARGS[@]}" \
#   --epochs 40 \
#   --train_mb_size "${BEST}" \
#   --job_name train_sequential_mode_mb${BEST}