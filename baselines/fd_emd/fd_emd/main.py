from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .algorithm import Client, FDEMD, FDEMDConfig
from .objectives import create_objectives


WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
EXPERIMENT_NAME_WANDB = os.environ.get("WANDB_PROJECT", "FD-EMD")
CSV_OUTPUT_DIR = Path("logs")
FIXED_DIMENSION_TASKS = {"branin_shifted", "hartmann6d_shifted", "cartpole"}


def regret_csv_filename(task: str, dim: int, seed: int | None = None) -> str:
    """Return the result filename used by the shared FedBO data directory."""
    dimension_suffix = "" if task in FIXED_DIMENSION_TASKS else f"{dim}d"
    seed_suffix = "" if seed is None else f"_{seed}"
    return f"{task}{dimension_suffix}_FD_EMD{seed_suffix}.csv"


def save_regret_csv(
    runs: list[list[Client]],
    task: str,
    dim: int,
    initial_points: int,
    output_dir: Path = CSV_OUTPUT_DIR,
    seed: int | None = None,
) -> Path:
    """Save the across-run mean-client simple-regret curve."""
    if not runs or not runs[0]:
        raise ValueError("At least one completed run with one client is required.")
    if initial_points < 1:
        raise ValueError("initial_points must be positive.")

    num_clients = len(runs[0])
    num_steps = len(runs[0][0].y) - initial_points
    if num_steps < 0:
        raise ValueError("A client has fewer observations than initial_points.")
    for run in runs:
        if len(run) != num_clients:
            raise ValueError("Every seed run must contain the same number of clients.")
        if any(len(client.y) - initial_points != num_steps for client in run):
            raise ValueError("Every client must contain the same number of regret steps.")

    run_regrets = np.empty((len(runs), num_steps), dtype=np.float64)
    for run_id, clients in enumerate(runs):
        for step in range(num_steps):
            observation_count = initial_points + step + 1
            client_regrets = [
                np.min(client.y[:observation_count]) - client.objective.optimum
                for client in clients
            ]
            run_regrets[run_id, step] = np.mean(client_regrets)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / regret_csv_filename(task, dim, seed)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["regret_step", "mean", "std", "max", "min"])
        for step in range(num_steps):
            values = run_regrets[:, step]
            writer.writerow(
                [
                    step,
                    float(values.mean()),
                    float(values.std()),
                    float(values.max()),
                    float(values.min()),
                ]
            )
    print(f"Saved regret curve to {output_path}")
    return output_path


def warm_start_csv_filename(task: str, dim: int, seed: int) -> str:
    dimension_suffix = "" if task in FIXED_DIMENSION_TASKS else f"{dim}d"
    return f"{task}{dimension_suffix}_FD_EMD_warm_start_{seed}.csv"


def save_agent_regret_csv(
    clients: list[Client],
    task: str,
    dim: int,
    rounds: int,
    initial_points: int,
    seed: int,
    output_dir: Path = CSV_OUTPUT_DIR,
) -> Path:
    """Save each client's simple regret after every one of its acquisition steps.

    A warm-started client joined late and has fewer steps. Its rows begin at the
    round it joined, so regret_step refers to the same global round for all clients.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / warm_start_csv_filename(task, dim, seed)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["seed", "agent", "warm_start", "regret_step", "simple_regret"])
        for client in clients:
            num_steps = len(client.y) - initial_points
            first_step = rounds - num_steps
            for local_step in range(num_steps):
                observation_count = initial_points + local_step + 1
                regret = np.min(client.y[:observation_count]) - client.objective.optimum
                writer.writerow(
                    [
                        seed,
                        client.client_id,
                        int(first_step > 0),
                        first_step + local_step,
                        float(regret),
                    ]
                )
    print(f"Saved per-agent regret curves to {output_path}")
    return output_path


def print_final_results(runs: list[list[Client]]) -> None:
    """Print mean final task performance across all completed seed runs."""
    if not runs:
        raise ValueError("At least one completed run is required.")
    if any(len(run) != len(runs[0]) for run in runs):
        raise ValueError("Every seed run must contain the same number of tasks.")

    rows = []
    for task_id, reference_client in enumerate(runs[0]):
        task_clients = [run[task_id] for run in runs]
        if any(client.objective.name != reference_client.objective.name for client in task_clients):
            raise ValueError("Task ordering must be identical across seed runs.")
        best_values = np.array([np.min(client.y) for client in task_clients], dtype=np.float64)
        simple_regrets = np.array(
            [best - client.objective.optimum for best, client in zip(best_values, task_clients)],
            dtype=np.float64,
        )
        rows.append(
            (
                reference_client.client_id,
                reference_client.objective.name,
                float(best_values.mean()),
                float(simple_regrets.mean()),
                float(simple_regrets.std()),
            )
        )

    print(f"\nFinal performance per task over {len(runs)} seed(s)")
    print("-" * 110)
    print(
        f"{'Client':>6}  {'Task':<38}  {'Mean best objective':>20}"
        f"  {'Mean simple regret':>20}  {'Regret std':>14}"
    )
    print("-" * 110)
    for client_id, task_name, best_value_mean, regret_mean, regret_std in rows:
        print(
            f"{client_id:>6}  {task_name:<38}  {best_value_mean:>20.10g}"
            f"  {regret_mean:>20.10g}  {regret_std:>14.10g}"
        )
    print("-" * 110)
    mean_regret = float(np.mean([row[3] for row in rows]))
    print(f"{'Mean final simple regret across tasks and seeds':<88}  {mean_regret:>20.10g}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FD-EMD federated optimization baseline.")
    parser.add_argument("--task", default="manytask", choices=[
        "manytask", "combination", "ackley_shifted", "ackley", "branin_shifted",
        "hartmann6d_shifted", "cartpole", "rastrigin", "sphere", "griewank",
        "rosenbrock", "schwefel",
    ])
    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument("--num-clients", type=int, default=18)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--initial-points", type=int, default=50)
    parser.add_argument("--num-centers", type=int, default=None, help="Defaults to 2 * dim + 1.")
    parser.add_argument("--ensemble-size", type=int, default=20)
    parser.add_argument("--auxiliary-size", type=int, default=1000)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--lcb-weight", type=float, default=2.0)
    parser.add_argument("--mutation-probability", type=float, default=None)
    parser.add_argument("--mutation-scale", type=float, default=0.1)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--kmeans-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warm-start-round",
        type=int,
        default=None,
        help="Add one client that joins after this many rounds and save per-agent regrets.",
    )
    parser.add_argument("--no-heterogeneous-shifts", action="store_true")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Run without importing, initializing, or logging to W&B.",
    )
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "disabled"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CSV_OUTPUT_DIR,
        help=f"Folder where the result CSV is saved (default: {CSV_OUTPUT_DIR}).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    completed_runs: list[list[Client]] = []
    for run_id in range(args.num_runs):
        seed = args.seed + run_id
        config = FDEMDConfig(
            rounds=args.rounds,
            initial_points=args.initial_points,
            num_centers=args.num_centers,
            ensemble_size=args.ensemble_size,
            auxiliary_size=args.auxiliary_size,
            population_size=args.population_size,
            generations=args.generations,
            lcb_weight=args.lcb_weight,
            mutation_probability=args.mutation_probability,
            mutation_scale=args.mutation_scale,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            ridge=args.ridge,
            kmeans_iterations=args.kmeans_iterations,
            seed=seed,
            warm_start_round=args.warm_start_round,
        )
        objective_rng = np.random.default_rng(seed)
        objectives = create_objectives(
            args.task,
            args.dim,
            args.num_clients + int(args.warm_start_round is not None),
            objective_rng,
            heterogeneous_shifts=not args.no_heterogeneous_shifts,
        )
        logged_config = {
            **asdict(config),
            "algorithm": "FD-EMD",
            "task": args.task,
            "dim": args.dim,
            "num_clients": args.num_clients,
            "wandb_entity": WANDB_ENTITY,
            "experiment_name_wandb": EXPERIMENT_NAME_WANDB,
            "trainer": {"_target_": "fd_emd"},
            "model": {"_target_": "rbfn"},
        }
        run_name = args.run_name
        if run_name is None:
            run_name = f"fd-emd-{args.task}-{args.dim}d-seed-{seed}"
        elif args.num_runs > 1:
            run_name = f"{run_name}-seed-{seed}"
        optimizer = FDEMD(objectives, config)
        if args.no_wandb:
            clients = optimizer.run()
        else:
            from .wandb_logging import WandBLogger

            with WandBLogger(
                logged_config,
                entity=WANDB_ENTITY,
                project=EXPERIMENT_NAME_WANDB,
                name=run_name,
                mode=args.wandb_mode,
            ) as logger:
                clients = optimizer.run(logger)
        completed_runs.append(clients)
        if args.warm_start_round is None:
            save_regret_csv(
                [clients],
                task=args.task,
                dim=args.dim,
                initial_points=args.initial_points,
                output_dir=args.output_dir,
                seed=seed,
            )
        else:
            save_agent_regret_csv(
                clients,
                task=args.task,
                dim=args.dim,
                rounds=args.rounds,
                initial_points=args.initial_points,
                seed=seed,
                output_dir=args.output_dir,
            )
    print_final_results(completed_runs)


if __name__ == "__main__":
    main()
