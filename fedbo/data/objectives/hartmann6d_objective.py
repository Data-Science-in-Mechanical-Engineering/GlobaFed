import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class Hartmann6DObjective(BaseObjective):
    def __init__(self, input_dim: int):
        super().__init__(input_dim, -3.32237, 0)

    def __call__(self, X: Array):
        return hartmann6d_objective(X)

    @staticmethod
    def bounds():
        return (0, 1)


@jax.jit
def hartmann6d_objective(X: Array):
    """
    Minimum value -3.32237. Bounds: [0, 1]
    Args:
        X: Array of shape (6,).
    Returns:
        f_X: Array of shape (1,)
    """
    assert X.shape[0] == 6
    alpha = jnp.array([1.0, 1.2, 3.0, 3.2])
    A = jnp.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    P = 1e-4 * jnp.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    X = X[None, :]
    inner_sum = -jnp.sum(A * (X - P) ** 2, axis=1)
    result = -jnp.sum(alpha * jnp.exp(inner_sum), axis=0)
    return jnp.array([result])
