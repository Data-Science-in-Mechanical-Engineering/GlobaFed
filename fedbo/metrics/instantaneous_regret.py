from jaxtyping import Array

from fedbo.metrics.base_metric import BaseMetric


class InstantaneousRegret(BaseMetric):
    name = "instantaneous_regret"

    def __call__(self, values: Array, min_value: Array):
        """Computes the instantaneous regret."""
        iregret = values[-1] - min_value
        return iregret
