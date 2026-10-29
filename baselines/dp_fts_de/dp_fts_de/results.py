from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


FIXED_DIMENSION_TASKS = {"branin_shifted", "hartmann6d_shifted", "cartpole"}


def regret_csv_filename(task: str, dim: int) -> str:
    dimension_suffix = "" if task in FIXED_DIMENSION_TASKS else f"{dim}d"
    return f"{task}{dimension_suffix}_DP_FTS_DE.csv"


def objective_optimum(objective) -> float:
    if hasattr(objective, "best_y"):
        return float(np.asarray(objective.best_y).squeeze())
    if objective.__class__.__name__ == "CartpoleLQRObjective":
        return -500.0
    return 0.0


def save_regret_csv(
    runs: list[list[np.ndarray]],
    run_optima: list[list[float]],
    task: str,
    dim: int,
    initial_points: int,
    output_dir: Path,
) -> Path:
    if not runs or not runs[0]:
        raise ValueError("At least one completed run with one client is required.")
    if len(runs) != len(run_optima):
        raise ValueError("Every completed run must have a matching optimum list.")
    num_clients = len(runs[0])
    num_steps = len(runs[0][0]) - initial_points
    if num_steps < 0:
        raise ValueError("A client has fewer observations than initial_points.")

    regrets = np.empty((len(runs), num_steps), dtype=float)
    for run_id, (histories, optima) in enumerate(zip(runs, run_optima)):
        if len(histories) != num_clients or len(optima) != num_clients:
            raise ValueError("Every run must contain the same number of clients.")
        if any(len(history) - initial_points != num_steps for history in histories):
            raise ValueError("Every client must contain the same number of steps.")
        for step in range(num_steps):
            count = initial_points + step + 1
            regrets[run_id, step] = np.mean(
                [
                    np.min(history[:count]) - optimum
                    for history, optimum in zip(histories, optima)
                ]
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / regret_csv_filename(task, dim)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["regret_step", "mean", "std", "max", "min"])
        for step, values in enumerate(regrets.T):
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
