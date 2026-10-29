#!/usr/bin/env bash
# Submit every FMTBO experiment script in this directory to the scheduler.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

for script in run_*.sh; do
  [ -e "$script" ] || continue
  job_id=$(sbatch --parsable "$script")
  printf '%-28s -> %s\n' "$script" "$job_id"
done
