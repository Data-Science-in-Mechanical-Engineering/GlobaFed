# FMTBO baseline

Adapted implementation of *Federated Many-Task Bayesian Optimization*:
H. Zhu, X. Wang and Y. Jin, IEEE Transactions on Evolutionary Computation,
doi: [10.1109/TEVC.2023.3279775](https://doi.org/10.1109/TEVC.2023.3279775).

Use a separate Python 3.11 or 3.12 environment. From this directory:

```bash
python -m pip install -e .
python fedmany/main.py --help
python fedmany/main.py --task manytask --dim 50 --num-clients 18 \
  --rounds 60 --initial-points 50 --num-runs 1 --seed 0 --output-dir ./logs
```

Aggregate regret CSVs default to `./logs`. The `--save_results` option also
writes client histories under `results_fbo_*`. `fedmany/results_wanb.py` is a
legacy uploader for those histories: edit its relative input paths and benchmark
settings to match your run. It reads `WANDB_ENTITY` and `WANDB_PROJECT`, and
requires `WANDB_MODE=online` to upload.

The `run_*.sh` scripts accept a seed, for example `bash run_ackley_5d.sh 0`.
For Slurm, activate the baseline environment, set your own `SBATCH_ACCOUNT` and
`SBATCH_PARTITION`, then run `bash launch_all.sh` from this directory.
`START_SEED` and `NUM_SEEDS` control the seed range (defaults: 0 and 20).
