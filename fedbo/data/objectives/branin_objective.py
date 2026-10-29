import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class BraninObjective(BaseObjective):
    def __init__(self, input_dim: int = 2):
        super().__init__(input_dim, 0.397887, 308.13)

    def __call__(self, X: Array):
        return branin_objective(X)

    @staticmethod
    def bounds():
        return (0, 10)


@jax.jit
def branin_objective(X: Array):
    """
    Minimum value ≈ 0.397887 at three points in the domain.
    Bounds: x1 in [-5, 10], x2 in [0, 15]
    Args:
        X: Array of shape (2,).
    Returns:
        f_X: Array of shape (1,)
    """
    assert X.shape[0] == 2
    X1 = X[0]
    X2 = X[1]
    result = (
        (X2 - (5.1 / (4 * jnp.pi**2)) * X1**2 + (5 / jnp.pi) * X1 - 6) ** 2
        + 10 * (1 - (1 / (8 * jnp.pi))) * jnp.cos(X1)
        + 10
    )
    return jnp.array([result])
