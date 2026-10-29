from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .algorithm import Client, IAFFBO, IAFFBOConfig
from .objectives import create_objectives


WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
EXPERIMENT_NAME_WANDB = os.environ.get("WANDB_PROJECT", "IAF-FBO")
CSV_OUTPUT_DIR = Path("logs")
FIXED_DIMENSION_TASKS = {"branin_shifted", "hartmann6d_shifted", "cartpole"}


def regret_csv_filename(task: str, dim: int, seed: int | None = None) -> str:
    dimension_suffix = "" if task in FIXED_DIMENSION_TASKS else f"{dim}d"
    seed_suffix = "" if seed is None else f"_{seed}"
    return f"{task}{dimension_suffix}_IAF_FBO{seed_suffix}.csv"


def save_regret_csv(
    runs: list[list[Client]],
    task: str,
    dim: int,
    initial_points: int,
    output_dir: Path = CSV_OUTPUT_DIR,
    seed: int | None = None,
) -> Path:
    if not runs or not runs[0]:
        raise ValueError("At least one completed run with one client is required.")
    num_clients = len(runs[0])
    num_steps = len(runs[0][0].y) - initial_points
    for run in runs:
        if len(run) != num_clients:
            raise ValueError("Every seed run must contain the same number of clients.")
        if any(len(client.y) - initial_points != num_steps for client in run):
            raise ValueError("Every client must contain the same number of regret steps.")

    run_regrets = np.empty((len(runs), num_steps), dtype=np.float64)
    for run_id, clients in enumerate(runs):
        for step in range(num_steps):
            count = initial_points + step + 1
            run_regrets[run_id, step] = np.mean(
                [
                    np.min(client.y[:count]) - client.objective.optimum
                    for client in clients
                ]
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / regret_csv_filename(task, dim, seed)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["regret_step", "mean", "std", "max", "min"])
        for step, values in enumerate(run_regrets.T):
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
    return f"{task}{dimension_suffix}_IAF_FBO_warm_start_{seed}.csv"


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
    if not runs:
        raise ValueError("At least one completed run is required.")
    print(f"\nFinal performance per task over {len(runs)} seed(s)")
    print("-" * 110)
    print(
        f"{'Client':>6}  {'Task':<38}  {'Mean best objective':>20}"
        f"  {'Mean simple regret':>20}  {'Regret std':>14}"
    )
    print("-" * 110)
    mean_regrets = []
    for task_id, reference_client in enumerate(runs[0]):
        task_clients = [run[task_id] for run in runs]
        best_values = np.asarray([client.best_y for client in task_clients])
        regrets = np.asarray(
            [
                best - client.objective.optimum
                for best, client in zip(best_values, task_clients)
            ]
        )
        mean_regrets.append(float(regrets.mean()))
        print(
            f"{reference_client.client_id:>6}  {reference_client.objective.name:<38}"
            f"  {best_values.mean():>20.10g}  {regrets.mean():>20.10g}"
            f"  {regrets.std():>14.10g}"
        )
    print("-" * 110)
    print(
        f"{'Mean final simple regret across tasks and seeds':<88}"
        f"  {np.mean(mean_regrets):>20.10g}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the IAF-FBO federated Bayesian optimization baseline."
    )
    parser.add_argument(
        "--task",
        default="manytask",
        choices=[
            "manytask",
            "combination",
            "ackley_shifted",
            "ackley",
            "branin_shifted",
            "hartmann6d_shifted",
            "cartpole",
            "rastrigin",
            "sphere",
            "griewank",
            "rosenbrock",
            "schwefel",
        ],
    )
    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument("--num-clients", type=int, default=18)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--initial-points", type=int, default=50)
    parser.add_argument("--num-groups", type=int, default=3)
    parser.add_argument("--server-sample-size", type=int, default=50)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--acquisition", choices=["ucb", "lcb", "ei"], default="ucb")
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--classifier-weight-decay", type=float, default=0.0)
    parser.add_argument("--max-pairwise-examples", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--phi", type=float, default=0.1)
    parser.add_argument("--group-model-probability", type=float, default=0.5)
    parser.add_argument("--gp-noise", type=float, default=1e-6)
    parser.add_argument("--gp-restarts", type=int, default=0)
    parser.add_argument("--kmeans-restarts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warm-start-round",
        type=int,
        default=None,
        help="Add one client that joins after this many rounds and save per-agent regrets.",
    )
    parser.add_argument("--no-heterogeneous-shifts", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
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
        config = IAFFBOConfig(
            rounds=args.rounds,
            initial_points=args.initial_points,
            num_groups=args.num_groups,
            server_sample_size=args.server_sample_size,
            beta=args.beta,
            acquisition=args.acquisition,
            classifier_epochs=args.classifier_epochs,
            classifier_learning_rate=args.classifier_learning_rate,
            classifier_batch_size=args.classifier_batch_size,
            classifier_weight_decay=args.classifier_weight_decay,
            max_pairwise_examples=args.max_pairwise_examples,
            population_size=args.population_size,
            generations=args.generations,
            phi=args.phi,
            group_model_probability=args.group_model_probability,
            gp_noise=args.gp_noise,
            gp_restarts=args.gp_restarts,
            kmeans_restarts=args.kmeans_restarts,
            seed=seed,
            warm_start_round=args.warm_start_round,
        )
        objectives = create_objectives(
            args.task,
            args.dim,
            args.num_clients + int(args.warm_start_round is not None),
            np.random.default_rng(seed),
            heterogeneous_shifts=not args.no_heterogeneous_shifts,
        )
        logged_config = {
            **asdict(config),
            "algorithm": "IAF-FBO",
            "task": args.task,
            "dim": args.dim,
            "num_clients": args.num_clients,
            "trainer": {"_target_": "iaf_fbo"},
            "model": {"_target_": "pairwise_classifier"},
        }
        run_name = args.run_name or f"iaf-fbo-{args.task}-{args.dim}d-seed-{seed}"
        if args.run_name is not None and args.num_runs > 1:
            run_name = f"{run_name}-seed-{seed}"
        optimizer = IAFFBO(objectives, config)
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
        print("-------------------")

if __name__ == "__main__":
    main()
