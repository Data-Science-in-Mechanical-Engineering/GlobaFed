from __future__ import annotations
import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class RastriginObjective(BaseObjective):
    def __init__(self, input_dim: int):
        super().__init__(input_dim, 0, 46.21 * input_dim)

    def __call__(self, X: Array):
        return rastrigin_objective(X)

    @staticmethod
    def bounds():
        return (-5.12, 5.12)


@jax.jit
def rastrigin_objective(X: Array):
    """
    Minimum value 0.
    Bounds: [-5.12, 5.12].
    Args:
        X: Array of shape (input_dim,).
        a: Optional
        b: Optional
        c: Optional
    Returns:
        f_X: Array of shape (1,)
    """
    input_dim = X.shape[0]
    result = 10 * input_dim + jnp.sum(X**2 - 10 * jnp.cos(2 * jnp.pi * X))
    return jnp.array([result])
