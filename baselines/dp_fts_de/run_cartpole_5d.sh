#!/bin/bash
#SBATCH --job-name=dpfts_cartpole5d
#SBATCH --output=logs/dp_fts_de_cartpole5d_%j.out
#SBATCH --error=logs/dp_fts_de_cartpole5d_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00

set -euo pipefail

# Activate the appropriate environment before running or submitting this script.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

python dp_fts_de/main_dp_fts_de.py \
  --task cartpole \
  --dim 5 \
  --num-clients 10 \
  --rounds 60 \
  --initial-points 5 \
  --seed 0 \
  --num-runs 20 \
  --output-dir ./logs
