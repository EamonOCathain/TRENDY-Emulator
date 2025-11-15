#!/bin/bash
set -euo pipefail

sbatch base.sh
sbatch ensmean.sh
sbatch stable.sh
sbatch TL.sh