#!/usr/bin/env python3
"""Combine per-seed IAF-FBO and FD-EMD warm-start CSVs and compute seed statistics.

Every per-seed file (for example ackley5d_IAF_FBO_warm_start_3.csv) holds one
simple-regret row per agent and regret step. For each experiment this script writes

- <base>_warm_start.csv: the rows of all seeds (seed, agent, warm_start,
  regret_step, simple_regret);
- <base>_warm_start_stats.csv: mean, std, max, and min over seeds for every
  regret step, both per agent and for the "cold" and "warm" groups. A group's
  per-seed value is the mean simple regret of its agents in that seed.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


SEED_FILENAME = re.compile(
    r"^(?P<base>.+_(?:IAF_FBO|FD_EMD))_warm_start_(?P<seed>\d+)\.csv$"
)
ROW_COLUMNS = ("seed", "agent", "warm_start", "regret_step", "simple_regret")
STATS_COLUMNS = ("group", "regret_step", "mean", "std", "max", "min", "num_seeds")


def _read_seed_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or tuple(reader.fieldnames) != ROW_COLUMNS:
            raise ValueError(f"{path} must have the columns {', '.join(ROW_COLUMNS)}.")
        rows = [
            {
                "seed": int(row["seed"]),
                "agent": int(row["agent"]),
                "warm_start": int(row["warm_start"]),
                "regret_step": int(row["regret_step"]),
                "simple_regret": float(row["simple_regret"]),
            }
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path} does not contain any regret rows.")
    return rows


def _group_seed_values(rows: list[dict]) -> dict[tuple[str, int], dict[int, float]]:
    """Map (group, regret_step) to the group's value in each seed."""
    agent_values: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        groups = (f"agent_{row['agent']}", "warm" if row["warm_start"] else "cold")
        for group in groups:
            agent_values[(group, row["regret_step"])][row["seed"]].append(
                row["simple_regret"]
            )
    return {
        key: {seed: float(np.mean(values)) for seed, values in per_seed.items()}
        for key, per_seed in agent_values.items()
    }


def _group_order(group: str) -> tuple[int, int]:
    if group == "cold":
        return (0, 0)
    if group == "warm":
        return (1, 0)
    return (2, int(group.removeprefix("agent_")))


def aggregate_seed_group(seed_paths: list[Path], rows_path: Path, stats_path: Path) -> None:
    """Write the combined rows and the over-seed statistics of one experiment."""
    rows = [row for path in seed_paths for row in _read_seed_rows(path)]
    rows.sort(key=lambda row: (row["seed"], row["agent"], row["regret_step"]))

    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    seed_values = _group_seed_values(rows)
    with stats_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(STATS_COLUMNS)
        for group, step in sorted(seed_values, key=lambda key: (_group_order(key[0]), key[1])):
            values = np.asarray(list(seed_values[(group, step)].values()))
            writer.writerow(
                [
                    group,
                    step,
                    float(values.mean()),
                    float(values.std()),
                    float(values.max()),
                    float(values.min()),
                    len(values),
                ]
            )


def aggregate_directory(input_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Discover and aggregate every warm-start seed group in a directory."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in input_dir.glob("*.csv"):
        match = SEED_FILENAME.match(path.name)
        if match:
            groups[match.group("base")].append((int(match.group("seed")), path))

    if not groups:
        raise FileNotFoundError(f"No per-seed warm-start CSVs found in {input_dir}.")

    destination = output_dir or input_dir
    outputs = []
    for base, numbered_paths in sorted(groups.items()):
        numbered_paths.sort(key=lambda item: item[0])
        rows_path = destination / f"{base}_warm_start.csv"
        stats_path = destination / f"{base}_warm_start_stats.csv"
        aggregate_seed_group([path for _, path in numbered_paths], rows_path, stats_path)
        outputs.extend([rows_path, stats_path])
        print(f"{base}: {len(numbered_paths)} seeds")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the per-seed CSVs (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the combined CSVs (default: input directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = aggregate_directory(args.input_dir, args.output_dir)
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
