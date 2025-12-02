# ---- Shared config for predict jobs (defaults; can be overridden by env) ----

: "${JOB_NAME:=Stable_Emulator_carry_2000_2020}"
: "${CARRY_FORWARD_STATES:=True}"
: "${SEQUENTIAL_MONTHS:=True}"

# SCENARIO is intentionally a default, so per-scenario wrappers can override it.
: "${SCENARIO:=S3}"   # override via: sbatch --export=ALL,SCENARIO=S0 ...

: "${DEVICE:=cpu}"    # set "cuda" if you enable a GPU
: "${ILAMB_DIR_GLOBAL:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/Stable_2000_2020_carry/MODELS}"
: "${NUMBER_TILES:=12}"
: "${WEIGHTS:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/checkpoints/carry/32_year/checkpoints/best.pt}"

# Carrying / Nudging
: "${NUDGE_LAMBDA:=0}"
: "${NUDGE_MODE:=none}"

# Paths
: "${FORCING_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/zarrs/inference}"
: "${OUT_DIR:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/data/predictions/stable_no_carry}"

# Periods
: "${STORE_PERIOD:=2000-01-01:2020-12-31}"
: "${WRITE_PERIOD:=2000-01-01:2020-12-31}"

# Optional filters (leave empty if unused)
: "${EXCLUDE_VARS:=}"

# Python env
: "${CONDA_ENV:=trendy-gpu}"
: "${PYTHON:=/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/2.predict/predict.py}"