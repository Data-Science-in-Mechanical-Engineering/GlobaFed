from __future__ import annotations
from datetime import datetime
import jax.numpy as jnp
import wandb
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

from fedbo.logger.base_logger import BaseLogger
from fedbo.visualization.model import create_model_prediction_fig

if TYPE_CHECKING:
    from jaxtyping import Array
    from fedbo.models import BaseModel
    from fedbo.data import ObjectiveDataset


class WandBLogger(BaseLogger):
    def __init__(
        self, entity: str | None = None, project: str = "fedbo",
        name: str | None = None, mode: str = "disabled",
    ):
        """Initializes the WandB Logger.
        Args:
            entity: The WandB entity.
            project: The WandB project name.
            name: Optional run name.
            mode: W&B mode; network logging requires explicit opt-in.
        """
        super().__init__()
        self.run = wandb.init(entity=entity, project=project, name=name, mode=mode)
        wandb.define_metric("regret_step")
        # Agent specific
        wandb.define_metric("instantaneous_regret/*", step_metric="regret_step")
        wandb.define_metric("simple_regret/*", step_metric="regret_step")
        wandb.define_metric("cumulative_regret/*", step_metric="regret_step")
        # Overall Metrics
        # Instananeous
        wandb.define_metric("avg/instantaneous_regret", step_metric="regret_step")
        wandb.define_metric("sum/instantaneous_regret", step_metric="regret_step")
        wandb.define_metric("worst/instantaneous_regret", step_metric="regret_step")
        wandb.define_metric("best/instantaneous_regret", step_metric="regret_step")
        # Simple
        wandb.define_metric("avg/simple_regret", step_metric="regret_step")
        wandb.define_metric("sum/simple_regret", step_metric="regret_step")
        wandb.define_metric("worst/simple_regret", step_metric="regret_step")
        wandb.define_metric("best/simple_regret", step_metric="regret_step")
        # Cumulative
        wandb.define_metric("avg/cumulative_regret", step_metric="regret_step")
        wandb.define_metric("sum/cumulative_regret", step_metric="regret_step")
        wandb.define_metric("worst/cumulative_regret", step_metric="regret_step")
        wandb.define_metric("best/cumulative_regret", step_metric="regret_step")

    def log_config(self, config: dict):
        """Logs the configuration."""
        self.run.config.update(config)

    def log_metrics(self, metrics: dict):
        """Logs the metrics."""
        self.run.log(metrics)

    def log_overall_regret(self, regret_per_iteration: dict[str, dict[int, list]]):
        """Logs the overall regret per iteration."""
        for metric_key, step_dict in regret_per_iteration.items():
            for step_key, values in step_dict.items():
                values_arr = jnp.array(values)
                log_dict = {
                    f"avg/{metric_key}": values_arr.mean(),
                    f"sum/{metric_key}": values_arr.sum(),
                    f"worst/{metric_key}": values_arr.max(),
                    f"best/{metric_key}": values_arr.min(),
                    "regret_step": step_key,
                }
                self.log_metrics(log_dict)

    def log_acquisition(
        self,
        model: BaseModel,
        dataset: ObjectiveDataset,
        agent_id: int,
        x_next: Array,
        y_next: Array,
    ):
        """Logs the acquisition step's details."""
        if not dataset.X.shape[1] == 1:
            self.logger.warning("Logging the acquisition is only supported for 1D inputs!")
            return
        figure_name = f"acquisition/agent_{agent_id}"
        fig = create_model_prediction_fig(model, dataset, x_next, y_next)
        self.run.log({figure_name: fig})
        plt.close(fig)

    def log_dataset(self, dataset: ObjectiveDataset, agent_id: int):
        """Logs the dataset."""
        table_X = wandb.Table(
            data=dataset.X.tolist(),
            columns=[f"x{i}" for i in range(dataset.X.shape[1])],
        )
        table_FX = wandb.Table(
            data=dataset.f_X.tolist(),
            columns=[f"y{i}" for i in range(dataset.f_X.shape[1])],
        )
        self.run.log({f"Agent_{agent_id}/X": table_X, f"Agent_{agent_id}/F(X)": table_FX})

    def finish(self):
        """Logs the total time taken and finishes the WandB run."""
        total_time = datetime.now() - self.start_time
        self.run.log({"total_time": total_time.total_seconds() / 60})
        self.log_info(f"Finished optimization in {total_time}!")
        self.run.finish()
