import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class LevyObjective(BaseObjective):
    def __init__(self, input_dim: int):
        # max_y is an approximate value within the typical bounds for scaling purposes.
        super().__init__(input_dim, min_y=0.0, max_y=1e4)

    def __call__(self, X: Array) -> Array:
        return levy_objective(X)

    @staticmethod
    def bounds() -> tuple[float, float]:
        return (-10.0, 10.0)


@jax.jit
def levy_objective(X: Array) -> Array:
    """
    Lévy function.
    Minimum value 0 at (1, 1, ..., 1).
    Bounds: [-10, 10].
    Args:
        X: Array of shape (input_dim,).
    Returns:
        f_X: Array of shape (1,)
    """
    # Transformation w = 1 + (x - 1) / 4
    w = 1.0 + (X - 1.0) / 4.0

    # First term
    term1 = jnp.sin(jnp.pi * w[0]) ** 2

    # Middle summation term
    w_mid = w[:-1]
    term2 = jnp.sum(
        (w_mid - 1.0) ** 2 * (1.0 + 10.0 * jnp.sin(jnp.pi * w_mid + 1.0) ** 2)
    )

    # Last term
    w_last = w[-1]
    term3 = (w_last - 1.0) ** 2 * (1.0 + jnp.sin(2.0 * jnp.pi * w_last) ** 2)

    result = term1 + term2 + term3
    return jnp.array([result])
