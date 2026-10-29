#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs
python dp_fts_de/main_dp_fts_de.py \
  --task hartmann6d_shifted \
  --dim 6 \
  --num-clients 10 \
  --rounds 60 \
  --initial-points 6 \
  --seed 0 \
  --num-runs 20 \
  --output-dir ./logs
