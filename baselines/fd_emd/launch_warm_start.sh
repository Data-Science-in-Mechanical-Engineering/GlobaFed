#!/usr/bin/env bash
# Submit the warm-start experiments of the homogeneous benchmark.
# Each job runs SEEDS_PER_JOB consecutive seeds of one experiment.
#
#   NUM_SEEDS=20 START_SEED=0 ./launch_warm_start.sh [TASK:DIM ...]
#
# Afterwards, combine the per-seed CSVs and compute the statistics over seeds with
#   python ../aggregate_warm_start_csvs.py logs_warm_start
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs_warm_start

num_seeds="${NUM_SEEDS:-20}"
start_seed="${START_SEED:-0}"
seeds_per_job="${SEEDS_PER_JOB:-20}"
if ! [[ "$num_seeds" =~ ^[1-9][0-9]*$ && "$start_seed" =~ ^[0-9]+$ && "$seeds_per_job" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SEEDS and SEEDS_PER_JOB must be positive and START_SEED must be non-negative." >&2
  exit 2
fi

if (($#)); then
  experiments=("$@")
else
  experiments=(
    ackley:1 ackley:5 ackley:10
    combination:1 combination:5 combination:10
    branin_shifted:2 hartmann6d_shifted:6 cartpole:5
  )
fi

for experiment in "${experiments[@]}"; do
  task="${experiment%%:*}"
  dim="${experiment##*:}"
  for ((offset = 0; offset < num_seeds; offset += seeds_per_job)); do
    seed=$((start_seed + offset))
    count=$((num_seeds - offset < seeds_per_job ? num_seeds - offset : seeds_per_job))
    log_prefix="logs_warm_start/${task}${dim}d_seed${seed}"
    job_id=$(sbatch --parsable \
      --job-name="fdemd_ws_${task}${dim}d_s${seed}" \
      --output="${log_prefix}_%j.out" \
      --error="${log_prefix}_%j.err" \
      run_warm_start.sh "$task" "$dim" "$seed" "$count")
    printf '%-24s seeds=%s..%s -> %s\n' "$experiment" "$seed" "$((seed + count - 1))" "$job_id"
  done
done
