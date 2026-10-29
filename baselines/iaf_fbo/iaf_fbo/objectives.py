"""Compatibility import for the shared baseline objective definitions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[2] / "objectives.py"
_MODULE_NAME = "_fedbo_baseline_objectives"
_MODULE = sys.modules.get(_MODULE_NAME)
if _MODULE is None:
    _SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Cannot load shared baseline objectives from {_SOURCE}.")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_MODULE_NAME] = _MODULE
    _SPEC.loader.exec_module(_MODULE)

Objective = _MODULE.Objective
TASK_NAMES = _MODULE.TASK_NAMES
create_objectives = _MODULE.create_objectives

__all__ = ["Objective", "TASK_NAMES", "create_objectives"]
