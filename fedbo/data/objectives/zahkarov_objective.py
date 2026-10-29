import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class ZakharovObjective(BaseObjective):
    def __init__(self, input_dim: int):
        # max_y is an approximate value within the bounds and grows rapidly with dimension.
        super().__init__(input_dim, min_y=0.0, max_y=1e7)

    def __call__(self, X: Array) -> Array:
        return zakharov_objective(X)

    @staticmethod
    def bounds() -> tuple[float, float]:
        return (-5.0, 10.0)


@jax.jit
def zakharov_objective(X: Array) -> Array:
    """
    Zakharov function.
    Minimum value 0 at (0, 0, ..., 0).
    Bounds: [-5, 10].
    Args:
        X: Array of shape (input_dim,).
    Returns:
        f_X: Array of shape (1,)
    """
    input_dim = X.shape[0]

    # First term: sum(x_i^2)
    term1 = jnp.sum(X**2)

    # Second term calculation: sum(0.5 * i * x_i)
    indices = jnp.arange(1, input_dim + 1)
    term2_sum = jnp.sum(0.5 * indices * X)

    # Combine terms
    result = term1 + term2_sum**2 + term2_sum**4
    return jnp.array([result])
