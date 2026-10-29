#!/bin/bash
#SBATCH --job-name=fmtbo_manytask
#SBATCH --output=logs/fmtbo_manytask_%j.out
#SBATCH --error=logs/fmtbo_manytask_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00

set -euo pipefail

# Activate the appropriate environment before running or submitting this script.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

python fedmany/main.py \
  --task manytask \
  --dim 10 \
  --num-clients 18 \
  --rounds 60 \
  --initial-points 50 \
  --seed 0 \
  --num-runs 20 \
  --output-dir ./logs
