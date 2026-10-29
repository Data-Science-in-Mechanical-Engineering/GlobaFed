#!/bin/bash
#SBATCH --job-name=fmtbo_branin2d
#SBATCH --output=logs/fmtbo_branin2d_%j.out
#SBATCH --error=logs/fmtbo_branin2d_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00

set -euo pipefail

# Activate the appropriate environment before running or submitting this script.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

python fedmany/main.py \
  --task branin_shifted \
  --dim 2 \
  --num-clients 10 \
  --rounds 60 \
  --initial-points 2 \
  --seed 0 \
  --num-runs 20 \
  --output-dir ./logs
