#!/usr/bin/env bash
# Central config for prediction + export jobs.
# Values are only set if unset in the environment.

# ---- High-level run ID ----
: "${JOB_NAME:=transfer_learn/avh15c1_lai_new/no_carry/S3}"

# ---- Data & model ----
: "${FORCING_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference}"
: "${OUT_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/}"
: "${WEIGHTS:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/1.train/runs/saved_checkpoints/transfer_learning/avh15c1_lai/second_train_to_epoch_22/checkpoints/best.pt}"

# ---- Periods ----
: "${STORE_PERIOD:=1901-01-01:2023-12-31}"
: "${WRITE_PERIOD:=1901-01-01:2023-12-31}"

# ---- Scenario ----
: "${SCENARIO:=S3}"

# ---- Carry/nudge ----
: "${CARRY_FORWARD_STATES:=False}"
: "${SEQUENTIAL_MONTHS:=False}"
: "${NUDGE_LAMBDA:=0}"
: "${NUDGE_MODE:=none}"

# ---- TL (transfer learning) ----
# space-separated list for predict.py --tl_vars
: "${TL_VARS:=lai_avh15c1}"
: "${TL_INITIAL_STATE:=1982}"

# ---- Optional excludes (comma-separated) ----
# : "${EXCLUDE_VARS:=pre_rolling_mean,tmp_rolling_mean,spfh_rolling_mean,tmax_rolling_mean,tmin_rolling_mean}"

# ---- Slurm shapes ----
: "${PREDICT_ARRAY:=0-7}"
: "${EXPORT_ARRAY:=0-9}"

# ---- Partitions/resources ----
: "${GPU_PARTITION:=gpu}"
: "${CPU_PARTITION:=work}"

# Predict resources
: "${PREDICT_CPUS:=8}"
: "${PREDICT_MEM:=24G}"
: "${PREDICT_TIME:=3-00:00:00}"
: "${PREDICT_GRES:=gpu:1}"

# Export resources
: "${EXPORT_CPUS:=2}"
: "${EXPORT_MEM:=8G}"
: "${EXPORT_TIME:=08:00:00}"

# ---- Conda ----
: "${CONDA_SH:=/User/homes/ecathain/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=trendy-gpu}"

# ---- Logs ----
: "${PREDICT_LOG_DIR:=logs/predictions}"
: "${EXPORT_LOG_DIR:=predictions/logs/netcdfs}"