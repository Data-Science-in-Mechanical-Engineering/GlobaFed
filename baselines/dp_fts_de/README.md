# DP-FTS-DE baseline

Adapted code for *Differentially Private Federated Bayesian Optimization with
Distributed Exploration* (NeurIPS 2021).
[Original paper](https://daizhongxiang.github.io/papers/dp_fbo.pdf).

Use a separate environment for the legacy GPy stack. From this directory:

```bash
python -m pip install -e .
python dp_fts_de/main_dp_fts_de.py --help
python dp_fts_de/main_dp_fts_de.py --task ackley --dim 1 --num-clients 3 \
  --num-runs 1 --rounds 2 --initial-points 2 --seed 0 --output-dir ./logs
```

GPy supports an older NumPy/SciPy stack than the root JAX package. Install this
project independently; do not combine their environments. The default run does
not enable differential privacy; use `--with-dp` to enable the original DP
aggregation. See `--help` for privacy and exploration parameters.

Aggregate regret CSVs default to `./logs`. Use `--intermediate-dir` to retain
client pickle traces. `dp_fts_de/results_wanb.py` is a legacy trace uploader:
edit its relative input path and benchmark settings to match your traces. It
reads `WANDB_ENTITY` and `WANDB_PROJECT`, and requires `WANDB_MODE=online` to upload.

The `run_*.sh` scripts accept a seed, for example `bash run_ackley_5d.sh 0`.
For Slurm, activate the baseline environment, set your own `SBATCH_ACCOUNT` and
`SBATCH_PARTITION`, then run `bash launch_all.sh` from this directory.
`START_SEED` and `NUM_SEEDS` control the seed range (defaults: 0 and 20).
