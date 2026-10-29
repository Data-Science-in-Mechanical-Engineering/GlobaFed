from __future__ import annotations
import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class CosineObjective(BaseObjective):
    def __init__(self, input_dim: int):
        super().__init__(input_dim, -input_dim, input_dim)

    def __call__(self, X: Array):
        return cosine_objective(X)

    @staticmethod
    def bounds():
        return (-jnp.pi * 2, jnp.pi * 2)


@jax.jit
def cosine_objective(X: Array):
    """
    Minimum value -input_dim.
    Bounds: [-2*pi, 2*pi].
    Args:
        X: Array of shape (input_dim,).
    Returns:
        f_X: Array of shape (1,)
    """
    result = jnp.sum(jnp.cos(X) * ((0.1 / (2 * jnp.pi)) * jnp.abs(X) - 1))
    return jnp.array([result])
