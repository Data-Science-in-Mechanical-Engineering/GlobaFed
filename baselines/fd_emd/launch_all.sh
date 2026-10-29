#!/usr/bin/env bash
# Submit one FD-EMD Slurm job per experiment and seed.
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs

num_seeds="${NUM_SEEDS:-20}"
start_seed="${START_SEED:-0}"
if ! [[ "$num_seeds" =~ ^[1-9][0-9]*$ && "$start_seed" =~ ^[0-9]+$ ]]; then
  echo "NUM_SEEDS must be positive and START_SEED must be non-negative." >&2
  exit 2
fi

if (($#)); then
  scripts=("$@")
else
  scripts=(run_*.sh)
fi

for script in "${scripts[@]}"; do
  [ -e "$script" ] || continue
  experiment="${script#run_}"
  experiment="${experiment%.sh}"
  for ((offset = 0; offset < num_seeds; offset++)); do
    seed=$((start_seed + offset))
    log_prefix="logs/${script%.sh}_seed${seed}"
    job_id=$(sbatch --parsable \
      --job-name="fdemd_${experiment}_s${seed}" \
      --output="${log_prefix}_%j.out" \
      --error="${log_prefix}_%j.err" \
      "$script" "$seed")
    printf '%-28s seed=%-4s -> %s\n' "$script" "$seed" "$job_id"
  done
done
