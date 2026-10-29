from abc import ABC, abstractmethod
from jaxtyping import Array


class BaseMetric(ABC):
    name = "base_metric"

    @abstractmethod
    def __call__(self, values: Array, min_value: Array):
        """Computes the metric."""
        pass
