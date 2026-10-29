#!/bin/bash
#SBATCH --job-name=fdemd_ackley10d
#SBATCH --output=logs/fd_emd_ackley10d_%j.out
#SBATCH --error=logs/fd_emd_ackley10d_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=4:00:00

set -euo pipefail

# Activate the appropriate environment before running or submitting this script.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

seed="${1:?Usage: sbatch $0 SEED}"

python -m fd_emd.main \
  --task ackley \
  --dim 10 \
  --num-clients 10 \
  --rounds 60 \
  --initial-points 10 \
  --local-epochs 5 \
  --learning-rate 0.06 \
  --seed "$seed" \
  --num-runs 1 \
  --no-wandb \
  --output-dir ./logs
