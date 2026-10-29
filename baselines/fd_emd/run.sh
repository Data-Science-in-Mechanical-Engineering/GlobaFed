#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
seed="${1:?Usage: $0 SEED}"
python -m fd_emd.main \
  --task manytask \
  --dim 10 \
  --num-clients 18 \
  --rounds 60 \
  --initial-points 50 \
  --local-epochs 5 \
  --learning-rate 0.06 \
  --seed "$seed" \
  --num-runs 1 \
  --no-wandb
