# Efficient Knowledge Transfer in Federated Bayesian Optimization through Neural Network Surrogates

Code for our paper *Efficient Knowledge Transfer in Federated Bayesian
Optimization through Neural Network Surrogates* (NeurIPS 2026) by Alexander
Gräfe, Max Van Gemmeren, Paul Brunzema, and Sebastian Trimpe. See
[Citation](#citation) for the BibTeX entry.

The main implementation supports MLP and variational Bayesian last layer
(VBLL) models, personalized multi-layer heads, local and centralized comparisons,
and a local Gaussian process baseline. Experiments use Hydra YAML configurations.

## Entry points

| Entry point | Purpose |
| --- | --- |
| `python -m fedbo.main` | Run the main optimizer; defaults to `federated_mlp`. |
| [`fedbo/config/`](fedbo/config/) | Experiment presets, datasets, models, training, and acquisition settings. |
| [`scripts/run/`](scripts/run/) | One experiment per script; accepts additional Hydra overrides. |
| [`scripts/seeds/`](scripts/seeds/) | Sequential local seed sweeps (30 seeds by default). |
| [`scripts/gridsearch/`](scripts/gridsearch/) | Local hyperparameter sweeps. |
| [`baselines/`](baselines/) | Separate IAF-FBO, FD-EMD, FMTBO, and DP-FTS-DE implementations. |
| [`fedbo/results.ipynb`](fedbo/results.ipynb) | Download W&B results and produce comparison/ablation plots. |
| [`baselines/aggregate_seed_csvs.py`](baselines/aggregate_seed_csvs.py) | Aggregate baseline CSV files across seeds. |
| [`baselines/aggregate_warm_start_csvs.py`](baselines/aggregate_warm_start_csvs.py) | Aggregate warm-start CSV files across seeds. |
| [`tests/`](tests/) | Main-package checks and executable research examples. |

## Installation

Use Python **3.11 or 3.12**. From the repository root, create an environment and
install the main package:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Alternatively, with Poetry 2 installed, use `poetry install` and prefix commands
with `poetry run`. The root `poetry.lock` records the main dependency resolution.
The examples below use CPU JAX; a GPU is optional. The `critdd` plotting dependency
is installed from its public Git repository and requires Git/network access.

Install each baseline in a **separate environment** using its own `pyproject.toml`.
The older GP baselines have dependencies that differ from the main JAX package.

## Quick start

Run a small federated MLP experiment on 1D Ackley with two clients:

```bash
JAX_PLATFORMS=cpu python -m fedbo.main --config-name federated_mlp \
  dataset=ackley1d transformations=shift num_agents=2 \
  bayesopt.num_iterations=2 trainer.num_rounds=1 \
  trainer.train_cfg.num_epochs=1 trainer.train_head_cfg.num_epochs=1 \
  model.hidden_features=8 model.num_hidden_layers=1
```

W&B logging is **disabled by default**. No W&B account or scheduler is needed for
this command. A smoke run checks execution; it does not reproduce paper results.
Full experiment presets retain their larger training and optimization budgets.

To inspect a resolved configuration without running an experiment:

```bash
python -m fedbo.main --config-name federated_vbll --cfg job --resolve
```

The shell wrappers accept the same overrides:

```bash
bash scripts/run/federated_mlp.sh dataset=ackley5d seed=1
```

## Configure experiments

Select a preset with `--config-name`:

| Preset | Training / surrogate |
| --- | --- |
| `federated_mlp`, `federated_vbll` | Federated MLP or VBLL |
| `federated_mlh_mlp`, `federated_mlh_vbll` | Federated models with personalized multi-layer heads |
| `local_mlp`, `local_vbll`, `local_gp` | Independent optimization per client |
| `centralized_mlp`, `centralized_vbll` | Centralized training |
| `warm_start`, `warm_start_vbll` | Add a client during optimization (MLP or VBLL) |

Dataset and transformation configurations must be paired:

| `dataset` | `transformations` | Typical clients |
| --- | --- | --- |
| `ackley1d`, `ackley5d`, `ackley10d`, `ackley50d`, `branin` | `shift` | 10 |
| `hartmann6d` | `shift_hartmann6d` | 10 |
| `combination1d`, `combination5d`, `combination10d`, `cartpole` | `default` | 10 |
| `manytask10d`, `manytask50d` | `none` | 18 |

ManyTask `.mat` benchmark assets are included in `fedbo/data/mat_files/` and the
legacy baseline task directories. No external data path is required.

Examples:

```bash
python -m fedbo.main --config-name local_gp \
  dataset=hartmann6d transformations=shift_hartmann6d seed=0

python -m fedbo.main --config-name federated_vbll \
  dataset=manytask10d transformations=none num_agents=18 seed=0

python -m fedbo.main --config-name warm_start_vbll \
  dataset=ackley5d transformations=shift seed=0
```

Changing the dataset does not change a preset's W&B project label. Override
`logger.project` or set `WANDB_PROJECT` when organizing runs for another dataset.
Set `bayesopt.num_iterations`, `trainer.num_rounds`, and the trainer's epoch
settings to control experiment cost. Centralized presets use
`trainer.local_train_cfg` and `trainer.global_train_cfg`; local/federated presets
use `trainer.train_cfg`.

## Seeds, sweeps, and Slurm

Run a short seed sweep locally:

```bash
bash scripts/seeds/local_gp.sh 'seed=0,1,2' bayesopt.num_iterations=2
python -m fedbo.main --config-name federated_mlp +hydra=fl_mlp_gridsearch --cfg hydra
```

Seed/grid scripts use Hydra's local launcher by default. Inspect a sweep before
launching it: the supplied grids can create many experiments.

For Slurm, activate your environment first and supply your own account and
resource choices. Run from the repository root on a filesystem available to
compute nodes:

```bash
python -m fedbo.main --config-name local_gp --multirun +hydra=seeds \
  hydra/launcher=submitit_slurm \
  hydra.launcher.account=YOUR_ACCOUNT hydra.launcher.partition=YOUR_PARTITION \
  hydra.launcher.cpus_per_task=1 hydra.launcher.mem_gb=5 \
  hydra.launcher.timeout_min=300
```

No cluster account, partition, module stack, or personal virtual environment path
is configured in the repository. Load any modules required by your site before
submission.

## Logging and results

Hydra writes configuration snapshots and process logs under `outputs/` for a
single run and `multirun/` for sweeps. With W&B disabled, the main optimizer does
not persist metric histories as CSV. Enable local W&B recording to retain them:

```bash
WANDB_MODE=offline python -m fedbo.main --config-name federated_mlp
```

For online logging, authenticate with your own W&B account and provide its
settings in your shell:

```bash
wandb login
export WANDB_ENTITY=YOUR_ENTITY
export WANDB_PROJECT=YOUR_PROJECT
export WANDB_MODE=online
python -m fedbo.main --config-name federated_mlp
```

The results notebook reads `WANDB_ENTITY` and loads the benchmark project names
listed in its cells. Adjust that list to match your projects. Run the notebook
with the main package environment, from the repository root or `fedbo/`.
Plot-data CSV exports default to `results/simulation_results/`; set
`FEDBO_RESULTS_DIR` to choose another directory. Figures are saved relative to
the notebook's working directory.

## Baselines

Start with the baseline-specific READMEs for algorithm settings and assumptions:

| Baseline | Directory | Command from that directory |
| --- | --- | --- |
| IAF-FBO | [`baselines/iaf_fbo`](baselines/iaf_fbo/README.md) | `python -m iaf_fbo.main` |
| FD-EMD | [`baselines/fd_emd`](baselines/fd_emd/README.md) | `python -m fd_emd.main` |
| FMTBO | [`baselines/fmtbo`](baselines/fmtbo/README.md) | `python fedmany/main.py` |
| DP-FTS-DE | [`baselines/dp_fts_de`](baselines/dp_fts_de/README.md) | `python dp_fts_de/main_dp_fts_de.py` |

For example, in a separate baseline environment:

```bash
cd baselines/iaf_fbo
python -m pip install -e .
python -m iaf_fbo.main --task sphere --dim 2 --num-clients 3 \
  --num-runs 1 --rounds 2 --initial-points 5 --num-groups 2 \
  --server-sample-size 5 --classifier-epochs 2 \
  --population-size 8 --generations 2 --no-wandb --output-dir ./logs
```

Baseline CSVs default to `./logs` relative to the working directory and contain
`regret_step,mean,std,max,min`. IAF-FBO and FD-EMD save separate seed files;
aggregate them with `python ../aggregate_seed_csvs.py logs`. Warm-start runs use
`logs_warm_start/` in their supplied scripts and have a separate aggregation
helper.

Baseline `run_*.sh` scripts can run locally with `bash run_ackley_5d.sh 0` or be
submitted through `launch_all.sh`. The launchers require Slurm; set
`SBATCH_ACCOUNT` and `SBATCH_PARTITION` in your own environment. Activate the
appropriate baseline environment and run submission commands from that baseline's
directory. `NUM_SEEDS` and `START_SEED` control the submitted seed range.

## Tests

IAF-FBO and FD-EMD have pytest suites. From each baseline directory, install
`pytest` into that baseline's environment and run `python -m pytest tests -q`.
Most root `tests/` files are standalone research examples, some opening plots;
`pytest tests` is not a headless full-suite validation command.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{grafe2026efficient,
  title     = {Efficient Knowledge Transfer in Federated {B}ayesian Optimization through Neural Network Surrogates},
  author    = {Gr{\"a}fe, Alexander and Van Gemmeren, Max and Brunzema, Paul and Trimpe, Sebastian},
  booktitle = {To Appear in Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
