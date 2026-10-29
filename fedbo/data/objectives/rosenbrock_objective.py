import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class RosenbrockObjective(BaseObjective):
    def __init__(self, input_dim: int):
        assert (
            input_dim >= 2
        ), "Rosenbrock function is only defined for dimensions >= 2."
        # max_y is an approximate value within the typical bounds for scaling purposes.
        super().__init__(input_dim, min_y=0.0, max_y=4e5)

    def __call__(self, X: Array) -> Array:
        return rosenbrock_objective(X)

    @staticmethod
    def bounds() -> tuple[float, float]:
        return (-2.048, 2.048)


@jax.jit
def rosenbrock_objective(X: Array) -> Array:
    """
    Rosenbrock function.
    Minimum value 0 at (1, 1, ..., 1).
    Bounds: Typically [-5, 10].
    Args:
        X: Array of shape (input_dim,). Must have input_dim >= 2.
    Returns:
        f_X: Array of shape (1,)
    """
    x_i = X[:-1]
    x_next = X[1:]
    result = jnp.sum(100.0 * (x_next - x_i**2) ** 2 + (x_i - 1.0) ** 2)
    return jnp.array([result])
