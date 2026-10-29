from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import wandb

from .algorithm import Client


class WandBLogger:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        entity: str | None = None,
        project: str = "IAF-FBO",
        name: str | None = None,
        mode: str = "disabled",
    ):
        self.started_at = datetime.now()
        self.run = wandb.init(
            entity=entity or os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT", project),
            name=name,
            config=config,
            mode=mode,
        )
        self.run.define_metric("regret_step")
        for metric in ("instantaneous_regret", "simple_regret", "cumulative_regret"):
            self.run.define_metric(f"{metric}/*", step_metric="regret_step")
            for aggregate in ("avg", "sum", "worst", "best"):
                self.run.define_metric(
                    f"{aggregate}/{metric}", step_metric="regret_step"
                )
        self.run.define_metric("server/*", step_metric="regret_step")
        self.run.define_metric("classifier/*", step_metric="regret_step")

    def log_round(
        self, round_id: int, clients: list[Client], diagnostics: dict[str, float]
    ) -> None:
        per_metric: dict[str, list[float]] = {
            "instantaneous_regret": [],
            "simple_regret": [],
            "cumulative_regret": [],
        }
        log_data: dict[str, float | int] = {
            "regret_step": round_id,
            **diagnostics,
        }
        for client in clients:
            regrets = client.y - client.objective.optimum
            values = {
                "instantaneous_regret": float(regrets[-1]),
                "simple_regret": float(regrets.min()),
                "cumulative_regret": float(regrets.sum()),
            }
            log_data[f"server/group/agent_{client.client_id}"] = client.group_id
            for metric, value in values.items():
                log_data[f"{metric}/agent_{client.client_id}"] = value
                per_metric[metric].append(value)
        for metric, values in per_metric.items():
            array = np.asarray(values)
            log_data.update(
                {
                    f"avg/{metric}": float(array.mean()),
                    f"sum/{metric}": float(array.sum()),
                    f"worst/{metric}": float(array.max()),
                    f"best/{metric}": float(array.min()),
                }
            )
        self.run.log(log_data)

    def finish(self) -> None:
        elapsed_minutes = (datetime.now() - self.started_at).total_seconds() / 60.0
        self.run.log({"total_time": elapsed_minutes})
        self.run.finish()

    def __enter__(self) -> "WandBLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.finish()
