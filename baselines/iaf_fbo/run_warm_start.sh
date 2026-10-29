#!/bin/bash
#SBATCH --job-name=iaf_warm_start
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --time=3:00:00             # Time limit hrs:min:sec

# Warm-start experiment on the homogeneous benchmark: 10 clients start together,
# an 11th client joins after 25 of the 50 rounds. Submitted by launch_warm_start.sh.

set -euo pipefail

# Activate the appropriate environment before running or submitting this script.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

task="${1:?Usage: sbatch $0 TASK DIM FIRST_SEED [NUM_SEEDS]}"
dim="${2:?Usage: sbatch $0 TASK DIM FIRST_SEED [NUM_SEEDS]}"
first_seed="${3:?Usage: sbatch $0 TASK DIM FIRST_SEED [NUM_SEEDS]}"
num_seeds="${4:-1}"

python -m iaf_fbo.main \
  --task "$task" \
  --dim "$dim" \
  --num-clients 10 \
  --rounds 50 \
  --warm-start-round 25 \
  --initial-points "$dim" \
  --classifier-epochs 20 \
  --seed "$first_seed" \
  --num-runs "$num_seeds" \
  --no-wandb \
  --output-dir ./logs_warm_start
