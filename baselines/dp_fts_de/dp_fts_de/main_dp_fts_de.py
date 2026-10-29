from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bayesian_optimization_dp_fts_de import dp_fts_de
from objectives import TASK_NAMES, create_objectives
from results import objective_optimum, save_regret_csv
from util.partitions import (
    generate_bounds_based_partition_info,
    generate_partition_info,
)
from util.random_features import generate_random_features


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the DP-FTS-DE federated optimization baseline."
    )
    parser.add_argument("--task", default="manytask", choices=TASK_NAMES)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument(
        "--num-clients", "--total_clients", dest="num_clients", type=int, default=18
    )
    parser.add_argument(
        "--num-runs", "--num_runs", dest="num_runs", type=int, default=20
    )
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--initial-points", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-random-features", type=int, default=100)
    parser.add_argument("--target-random-features", type=int, default=100)
    parser.add_argument("--partitions-per-dimension", type=int, default=2)
    parser.add_argument("--gp-opt-schedule", type=int, default=5)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--warm-start-round", type=int, default=25)
    parser.add_argument(
        "--with-dp",
        action="store_true",
        help="Enable the original DP aggregation (the paper code defaults to no DP).",
    )
    parser.add_argument("--rdp", action="store_true")
    parser.add_argument("--sampling-probability", type=float, default=0.35)
    parser.add_argument("--sensitivity", type=float, default=22.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs"),
        help="Folder where the aggregate regret CSV is saved.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="Optionally save each client's original pickle trace here.",
    )
    return parser.parse_args(argv)


def _partition_info(args, tasks, num_agents):
    if args.task == "manytask":
        info = generate_bounds_based_partition_info(
            args.dim,
            [(task.x_lb, task.x_ub) for task in tasks],
            num_agents,
            T_max=args.rounds + 1,
        )
    else:
        info = generate_partition_info(
            args.dim,
            (tasks[0].x_lb, tasks[0].x_ub),
            num_agents,
            args.partitions_per_dimension,
            T_max=args.rounds + 1,
        )
    return (
        info["partitions"],
        info["partition_assignment"],
        info["all_weights"],
    )


def run_once(args, run_id):
    seed = args.seed + run_id
    np.random.seed(seed)
    num_agents = args.num_clients + int(args.warm_start)
    tasks = create_objectives(
        args.task,
        args.dim,
        num_agents,
        np.random.default_rng(seed),
    )
    partitions, assignments, all_weights = _partition_info(
        args, tasks, num_agents
    )
    num_partitions = len(partitions)
    random_features = generate_random_features(
        args.dim, args.num_random_features
    )
    probability = 1.0 - 1.0 / (np.arange(args.rounds + 6) + 1.0)

    if args.intermediate_dir is not None:
        trace_dir = args.intermediate_dir / f"run_{run_id}"
        trace_dir.mkdir(parents=True, exist_ok=True)
    else:
        trace_dir = None

    def objective(x, client_id):
        return -float(tasks[client_id](np.asarray(x)))

    def create_optimizer(client_id):
        log_file = (
            str(trace_dir / f"agent_{client_id}.p")
            if trace_dir is not None
            else None
        )
        bounds = (tasks[client_id].x_lb, tasks[client_id].x_ub)
        return dp_fts_de(
            f=objective,
            pbounds={f"x{i}": bounds for i in range(1, args.dim + 1)},
            gp_opt_schedule=args.gp_opt_schedule,
            log_file=log_file,
            M=args.num_random_features,
            N=num_agents,
            random_features=random_features,
            pt=probability,
            M_target=args.target_random_features,
            partitions=partitions,
            partition_assignment=assignments[client_id],
        )

    primary_ids = list(range(args.num_clients))
    optimizers = [create_optimizer(client_id) for client_id in primary_ids]
    warm_optimizer = create_optimizer(args.num_clients) if args.warm_start else None
    active_ids = primary_ids.copy()
    all_w_t = None

    # Iteration zero initializes each GP; the remaining iterations are BO rounds.
    for iteration in range(args.rounds + 1):
        weights = []
        for client_id in active_ids:
            weights.append(
                optimizers[client_id].maximize(
                    n_iter=args.rounds,
                    init_points=args.initial_points,
                    all_w_t=all_w_t,
                    init_flag=iteration == 0,
                    iter_fed=iteration,
                    agent_ind=client_id,
                )
            )
        all_w_nt = np.asarray(weights).reshape(len(active_ids), -1)

        if (
            warm_optimizer is not None
            and iteration + 1 == args.warm_start_round
        ):
            warm_optimizer.maximize(
                n_iter=args.rounds,
                init_points=args.initial_points,
                all_w_t=all_w_t,
                init_flag=True,
                iter_fed=iteration,
                agent_ind=args.num_clients,
            )
            optimizers.append(warm_optimizer)
            active_ids.append(args.num_clients)

        all_w_t = []
        if not args.with_dp:
            for partition_id in range(num_partitions):
                client_weights = all_weights[iteration][partition_id]
                aggregate = np.zeros(args.num_random_features)
                for row, client_id in enumerate(active_ids[: len(all_w_nt)]):
                    aggregate += client_weights[client_id] * all_w_nt[row]
                all_w_t.append(aggregate)
            continue

        if args.rdp:
            sample_size = int(num_agents * args.sampling_probability)
            selected = list(
                np.random.choice(active_ids, sample_size, replace=False)
            )
        else:
            selected = [
                client_id
                for client_id in active_ids
                if np.random.random() < args.sampling_probability
            ]
        for partition_id in range(num_partitions):
            if not selected:
                all_w_t.append(np.random.random(args.num_random_features))
                continue
            client_weights = all_weights[iteration][partition_id]
            partition_sensitivity = args.sensitivity / np.sqrt(num_partitions)
            aggregate = np.zeros(args.num_random_features)
            for client_id in selected:
                if client_id not in active_ids[: len(all_w_nt)]:
                    continue
                row = active_ids.index(client_id)
                vector = all_w_nt[row]
                vector = vector / max(
                    1.0, np.linalg.norm(vector) / partition_sensitivity
                )
                aggregate += (
                    client_weights[client_id]
                    * vector
                    / args.sampling_probability
                )
            noise = np.random.normal(
                0.0,
                np.max(client_weights)
                * args.noise_multiplier
                * args.sensitivity
                / args.sampling_probability,
                args.num_random_features,
            )
            all_w_t.append(aggregate + noise)

    # DP-FTS-DE maximizes the negated objective, so convert traces back here.
    histories = [
        -np.asarray(optimizer.Y, dtype=float).copy()
        for optimizer in optimizers[: args.num_clients]
    ]
    optima = [
        objective_optimum(task) for task in tasks[: args.num_clients]
    ]
    return histories, optima


def main(argv=None):
    args = parse_args(argv)
    completed_runs = []
    run_optima = []
    for run_id in range(args.num_runs):
        histories, optima = run_once(args, run_id)
        completed_runs.append(histories)
        run_optima.append(optima)
    save_regret_csv(
        completed_runs,
        run_optima,
        task=args.task,
        dim=args.dim,
        initial_points=args.initial_points,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
