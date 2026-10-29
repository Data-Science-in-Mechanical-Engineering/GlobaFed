import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class GriewankObjective(BaseObjective):
    def __init__(self, input_dim: int):
        super().__init__(input_dim, 0, 91 * input_dim)

    def __call__(self, X: Array):
        return griewank_objective(X)

    @staticmethod
    def bounds():
        return (-600, 600)


@jax.jit
def griewank_objective(X: Array):
    """
    Minimum value 0.
    Bounds: [-600, 600].
    Args:
        X: Array of shape (input_dim,).
    Returns:
        f_X: Array of shape (1,)
    """
    sum_term = jnp.sum(X**2) / 4000.0
    prod_term = jnp.prod(jnp.cos(X / jnp.sqrt(jnp.arange(1, X.shape[0] + 1))))
    result = sum_term - prod_term + 1
    return jnp.array([result])
