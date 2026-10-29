import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class AckleyObjective(BaseObjective):
    def __init__(self, input_dim: int):
        super().__init__(input_dim, 0, 22.35)

    def __call__(self, X: Array) -> Array:
        return ackley_objective(X)

    @staticmethod
    def bounds():
        return (-5, 10)


@jax.jit
def ackley_objective(
    X: Array, a: float = 20.0, b: float = 0.2, c: float = 2.0 * jnp.pi
):
    """
    Minimum value 0.
    Bounds: [-5, 10].
    Args:
        X: Array of shape (input_dim,).
        a: Optional
        b: Optional
        c: Optional
    Returns:
        f_X: Array of shape (1,)
    """
    input_dim = X.shape[0]
    result = (
        -a * jnp.exp(-b * jnp.linalg.norm(X, ord=2) * jnp.sqrt(1.0 / input_dim))
        - jnp.exp(1.0 / input_dim * jnp.sum(jnp.cos(c * X)))
        + a
        + jnp.exp(1.0)
    )
    return jnp.array([result])
