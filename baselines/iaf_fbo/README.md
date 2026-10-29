# IAF-FBO baseline

This directory contains a reproducible implementation of **Implicit
Acquisition Function Federated Bayesian Optimization (IAF-FBO)**.

Each client keeps its observations private, fits an isotropic RBF-kernel Gaussian
process by marginal-likelihood maximization, and converts acquisition values
into pairwise labels. A PyTorch MLP with widths `1.5d`, `d`, and `0.5d`
learns those comparisons. The server clusters flattened classifier parameters
with k-means, averages parameters within each group, and generates one
personalized proposal per client with the modified competitive swarm optimizer.

The implementation preserves the specification's communication boundary:
only classifier parameters are aggregated. Observations, objective values,
acquisition values, GPs, and pairwise datasets stay local.

## Explicit choices for unspecified details

- The default server sample size is 50 and samples are uniform in `[0, 1]^d`.
- All objectives are minimized, while the classifier and CSO treat larger
  scores as better. UCB is therefore applied to utility `-f`, LCB on `f` is
  negated, and EI is used directly. Pair labels follow Algorithm 2: a label is
  one exactly when the first utility is greater.
- Both orders of every unordered pair are used, including ties.
  `--max-pairwise-examples` can bound memory for large experiments.
- Hidden widths use Python's nearest-integer `round`, with a minimum of one.
- The MLP uses ReLU, binary cross entropy, and mini-batch Adam.
- CSO velocities start at zero, positions are clipped to `[0, 1]^d`, and a
  shuffled knockout tournament selects the final champion.
- All clients participate in every round.

Defaults from the paper are used where reported: 60 rounds, 3 groups, a
100-particle swarm, 300 CSO generations, `phi=0.1`, and equal local/group
search-model probability. Values not reported by the paper remain CLI options.

The benchmark objective adapters are shared with `fd_emd`, ensuring both
baselines use identical normalized domains, task shifts, optima, and ManyTask
data.

## Run

Install this baseline in its own Python 3.11 or 3.12 environment (`python -m pip install -e .`).
From this directory:

```bash
python -m iaf_fbo.main \
  --task manytask \
  --dim 50 \
  --num-clients 18 \
  --rounds 60 \
  --initial-points 50
```

A small offline smoke run:

```bash
python -m iaf_fbo.main --task sphere --dim 2 --num-clients 3 \
  --num-runs 1 --rounds 2 --initial-points 5 --num-groups 2 \
  --server-sample-size 5 --classifier-epochs 2 \
  --population-size 8 --generations 2 --no-wandb --output-dir /tmp
```

Results use the same W&B regret keys as the other baselines and are saved to
`./logs` by default, one file per seed, for example
`ackley5d_IAF_FBO_0.csv`. Use `--output-dir` to save the CSV elsewhere.

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
