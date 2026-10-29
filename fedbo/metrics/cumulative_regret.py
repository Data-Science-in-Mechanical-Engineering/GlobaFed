import jax.numpy as jnp
from jaxtyping import Array

from fedbo.metrics.base_metric import BaseMetric


class CumulativeRegret(BaseMetric):
    name = "cumulative_regret"

    def __call__(self, values: Array, min_value: Array):
        """Computes the cumulative regret."""
        iregrets = values - min_value
        return jnp.sum(iregrets)
