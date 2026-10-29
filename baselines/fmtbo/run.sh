#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
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
