#!/usr/bin/env python3
"""Combine per-seed IAF-FBO and FD-EMD CSVs into the legacy CSV format."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


SEED_FILENAME = re.compile(
    r"^(?P<base>.+_(?:IAF_FBO|FD_EMD))_(?P<seed>\d+)\.csv$"
)
OUTPUT_COLUMNS = ("regret_step", "mean", "std", "max", "min")


def _read_seed_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not {"regret_step", "mean"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"{path} must contain at least the regret_step and mean columns."
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} does not contain any regret rows.")
    steps = [row["regret_step"] for row in rows]
    try:
        values = np.asarray([float(row["mean"]) for row in rows], dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{path} contains a non-numeric mean value.") from error
    return steps, values


def aggregate_seed_group(seed_paths: list[Path], output_path: Path) -> Path:
    """Aggregate one experiment's per-seed files into its legacy result file."""
    if not seed_paths:
        raise ValueError("At least one per-seed CSV is required.")

    reference_steps: list[str] | None = None
    curves = []
    for path in seed_paths:
        steps, values = _read_seed_csv(path)
        if reference_steps is None:
            reference_steps = steps
        elif steps != reference_steps:
            raise ValueError(f"Regret steps in {path} do not match the other seeds.")
        curves.append(values)

    assert reference_steps is not None
    seed_curves = np.vstack(curves)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(OUTPUT_COLUMNS)
        for step, values in zip(reference_steps, seed_curves.T):
            writer.writerow(
                [
                    step,
                    float(values.mean()),
                    float(values.std()),
                    float(values.max()),
                    float(values.min()),
                ]
            )
    return output_path


def aggregate_directory(input_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Discover and aggregate every IAF-FBO/FD-EMD seed group in a directory."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in input_dir.glob("*.csv"):
        match = SEED_FILENAME.match(path.name)
        if match:
            groups[match.group("base")].append((int(match.group("seed")), path))

    if not groups:
        raise FileNotFoundError(f"No per-seed baseline CSVs found in {input_dir}.")

    destination = output_dir or input_dir
    outputs = []
    for base, numbered_paths in sorted(groups.items()):
        numbered_paths.sort(key=lambda item: item[0])
        output_path = destination / f"{base}.csv"
        aggregate_seed_group([path for _, path in numbered_paths], output_path)
        outputs.append(output_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate files such as ackley5d_IAF_FBO_0.csv and "
            "ackley5d_IAF_FBO_1.csv into the legacy ackley5d_IAF_FBO.csv."
        )
    )
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
        help="Directory for aggregate CSVs (default: input directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = aggregate_directory(args.input_dir, args.output_dir)
    for output in outputs:
        print(f"Saved aggregate CSV to {output}")


if __name__ == "__main__":
    main()
