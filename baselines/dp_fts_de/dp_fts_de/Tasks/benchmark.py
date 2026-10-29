"""Compatibility adapters for the shared baseline objective factory."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from objectives import TASK_NAMES, create_objectives


def create_tasks(task_name: str, num_tasks: int, dim: int):
    return create_objectives(
        task_name,
        dim,
        num_tasks,
        np.random.default_rng(np.random.randint(0, 2**32)),
    )


def create_tasks_diff_func(dim: int, normalized: bool = False):
    del normalized
    return create_tasks("manytask", 18, dim)


__all__ = ["TASK_NAMES", "create_tasks", "create_tasks_diff_func"]
