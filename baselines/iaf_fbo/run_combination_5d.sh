#!/bin/bash
#SBATCH --job-name=iaf_comb5d         # Job name
#SBATCH --output=logs/iaf_fbo_combination5d_%j.out  # Standard output and error log
#SBATCH --error=logs/iaf_fbo_combination5d_%j.err   # Error log
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=24           # Number of CPU cores per task
#SBATCH --time=4:00:00             # Time limit hrs:min:sec

set -euo pipefail

# Activate the appropriate environment before running or submitting this script.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

seed="${1:?Usage: sbatch $0 SEED}"

python -m iaf_fbo.main \
  --task combination \
  --dim 5 \
  --num-clients 10 \
  --rounds 60 \
  --initial-points 5 \
  --classifier-epochs 20 \
  --seed "$seed" \
  --num-runs 1 \
  --no-wandb \
  --output-dir ./logs
