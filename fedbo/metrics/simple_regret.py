import jax.numpy as jnp
from jaxtyping import Array

from fedbo.metrics.base_metric import BaseMetric


class SimpleRegret(BaseMetric):
    name = "simple_regret"

    def __call__(self, values: Array, min_value: Array):
        """Computes the simple regret."""
        iregrets = values - min_value
        return jnp.min(iregrets)
