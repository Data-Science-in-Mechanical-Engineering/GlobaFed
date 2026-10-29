from __future__ import annotations
from datetime import datetime
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyping import Array
    from fedbo.models import PredictionModel
    from fedbo.data import ObjectiveDataset


class BaseLogger:
    def __init__(self):
        self.start_time = datetime.now()
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def log_info(self, message: str):
        """Logs an info message."""
        self.logger.info(message)

    def log_config(self, config: dict):
        """Logs the configuration."""
        self.log_info(f"Config:\n{config}")

    def log_metrics(self, metrics: dict):
        """Logs the metrics."""
        self.log_info(f"Metrics:\n{metrics}")

    def log_overall_regret(self, regret_per_iteration: dict[str, dict[int, list]]):
        """Logs the overall regret per iteration."""
        self.log_info(f"Overall Regrets:\n{regret_per_iteration}")

    def log_acquisition(
        self,
        model: PredictionModel,
        dataset: ObjectiveDataset,
        agent_id: int,
        x_next: Array,
        y_next: Array,
    ):
        """Logs the acquisition step's details."""
        self.logger.info("Log Acquisition")

    def log_dataset(self, dataset: ObjectiveDataset, agent_id: int):
        """Logs the dataset."""
        self.logger.info("Log Dataset")

    def finish(self):
        """Logs the total time taken."""
        total_time = datetime.now() - self.start_time
        self.log_info(f"Finished optimization in {total_time}!")
