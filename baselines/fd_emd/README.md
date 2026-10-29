# FD-EMD baseline

This directory contains a NumPy implementation of **Federated Data-Driven
Optimization with Ensemble Model Distillation (FD-EMD)**.

It implements:

- local Gaussian radial-basis-function networks with `2d + 1` centers found by
  k-means and weights initialized using the paper's least-squares rule;
- five full-batch L2 update epochs with learning rate `0.06` for client and
  student updates, recalculating RBF centers with k-means on each updated
  training dataset;
- LCB acquisition from the mean and standard deviation of local and global RBFN
  predictions;
- an evolutionary search with elitism, blend crossover, and Gaussian mutation;
- server-side diagonal-Gaussian model aggregation, ensemble sampling,
  pseudo-label generation, and RBFN student distillation;
- optional per-round W&B logging, using `WANDB_ENTITY` and `WANDB_PROJECT`
  from your environment and the same regret keys as the main project.
- a final console table aggregating all requested seeds, with the mean best
  objective, mean simple regret, and regret standard deviation for every client
  task, followed by the overall mean across tasks and seeds.

All decision vectors are represented in `[0, 1]^d` inside the optimizer and
mapped to each client's native bounds only for objective evaluation. The raw
RBF parameters (centers, widths, and output weights) are serialized when the
server constructs the diagonal Gaussian from equation (7); no output
normalization or logarithmic reparameterization is inserted. Before aggregation,
each local model's nodes are sorted by the squared-sum matching metric
`M_j = sum_i(c[j, i]^2)`, with centers, widths, and output weights reordered
together. The server initializes a random `w_0^g`, and ensemble distillation starts only after
the first client round, matching Algorithms 1 and 3.

## Run

Install this baseline in its own Python 3.11 or 3.12 environment (`python -m pip install -e .`).
From this directory:

```bash
python -m fd_emd.main \
  --task manytask \
  --dim 50 \
  --num-clients 18 \
  --rounds 60 \
  --initial-points 50
```

For a quick local smoke run without importing, initializing, or logging in to
W&B:

```bash
python -m fd_emd.main --task sphere --dim 2 --num-clients 3 \
  --num-runs 1 --rounds 2 --initial-points 5 --num-centers 5 \
  --ensemble-size 3 --auxiliary-size 20 \
  --population-size 8 --generations 2 --no-wandb \
  --output-dir ./results
```

Use `python -m fd_emd.main --help` for all settings. The paper defaults are
configured for 20 runs, initial points, communication rounds, local epochs,
learning rate, RBF centers, EA population and generations, and the LCB weight.
The paper does not specify the ensemble size, auxiliary-set size, or detailed
EA crossover and mutation operators; those remain explicit implementation
choices.

The bundled `manytask` option implements the paper's 18-task FD-MOP1 setup.
The other task factories from `fmtbo/fedmany/Tasks/benchmark.py` are also
available as `ackley`, `ackley_shifted`, `branin_shifted`,
`hartmann6d_shifted`, `combination`, and `cartpole`. Branin, Hartmann-6, and
CartPole require `--dim 2`, `--dim 6`, and `--dim 5`, respectively.
FD-MOP2's rotation/partition instances and the landmine dataset are not shipped
with this repository and therefore are not presented as reproducible here.

FD-EMD writes one CSV per seed to `./logs` by default. Use `--output-dir`
to save it elsewhere. Each CSV contains `regret_step,mean,std,max,min` and
appends its seed to the previous filename, for example
`ackley10d_FD_EMD_0.csv` or `cartpole_FD_EMD_0.csv`.

W&B is disabled by default. Set `WANDB_ENTITY`, `WANDB_PROJECT`, and
`WANDB_MODE=online` to enable it (or pass `--wandb-mode online`).
Activate your environment before submitting jobs. Supply your site settings
through `SBATCH_ACCOUNT` and `SBATCH_PARTITION`; no account is bundled here.

Submit one Slurm job for each of the 20 default seeds and every bundled
experiment with:

```bash
./launch_all.sh
```

The launcher can target selected experiments and seed ranges without creating
additional scripts. For example, this submits seeds 5 through 7 for Ackley-5D:

```bash
START_SEED=5 NUM_SEEDS=3 ./launch_all.sh run_ackley_5d.sh
```

After the seed jobs have finished, recreate the old aggregate CSV (and any
other complete seed groups in the directory) with:

```bash
python ../aggregate_seed_csvs.py logs
```

## Test

```bash
pytest
```
