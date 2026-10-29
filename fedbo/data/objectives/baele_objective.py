import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class BealeObjective(BaseObjective):
    def __init__(self, input_dim: int):
        assert input_dim == 2, "Beale function is only defined for 2 dimensions."
        # max_y is an approximate value within the bounds for scaling.
        super().__init__(input_dim, min_y=0.0, max_y=2e5)

    def __call__(self, X: Array) -> Array:
        return beale_objective(X)

    @staticmethod
    def bounds() -> tuple[float, float]:
        return (-4.5, 4.5)


@jax.jit
def beale_objective(X: Array) -> Array:
    """
    Beale function. Strictly 2D.
    Minimum value 0 at (3, 0.5).
    Bounds: [-4.5, 4.5].
    Args:
        X: Array of shape (2,).
    Returns:
        f_X: Array of shape (1,)
    """
    x, y = X[0], X[1]
    term1 = (1.5 - x + x * y) ** 2
    term2 = (2.25 - x + x * y**2) ** 2
    term3 = (2.625 - x + x * y**3) ** 2
    result = term1 + term2 + term3
    return jnp.array([result])
