import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class SphereObjective(BaseObjective):
    def __init__(self, input_dim: int):
        super().__init__(input_dim, 0, 5.12**2 * input_dim)

    def __call__(self, X: Array):
        return sphere_objective(X)

    @staticmethod
    def bounds():
        return (-5.12, 5.12)


@jax.jit
def sphere_objective(X: Array):
    """
    Minimum value 0.
    Bounds: [-5.12, 5.12].
    Args:
        X: Array of shape (input_dim,).
    Returns:
        f_X: Array of shape (1,)
    """
    result = jnp.sum(X**2)
    return jnp.array([result])
