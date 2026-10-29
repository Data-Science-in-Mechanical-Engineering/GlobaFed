import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class SchwefelObjective(BaseObjective):
    """
    Schwefel benchmark function.

    This function is complex, with many local minima. The global minimum is
    located far from the next-best local minimum, making it challenging for
    optimization algorithms.
    """

    def __init__(self, input_dim: int):
        # The maximum value is approximated. It occurs when each x_i sin(sqrt(|x_i|))
        # term is at its minimum (~-450), leading to a max value of ~900 * input_dim.
        super().__init__(input_dim, 0, 950 * input_dim)

    def __call__(self, X: Array) -> Array:
        """Evaluates the Schwefel function for a given input array."""
        return schwefel_objective(X)

    @staticmethod
    def bounds() -> tuple[int, int]:
        """Returns the recommended search space bounds for the function."""
        return (-500, 500)


@jax.jit
def schwefel_objective(X: Array) -> Array:
    """
    JAX-jitted implementation of the Schwefel function.

    The formula is: f(x) = 418.9829 * d - sum(x_i * sin(sqrt(|x_i|)))
    Global minimum is f(x) = 0 at x_i = 420.9687 for all i.

    Args:
        X: Array of shape (input_dim,).

    Returns:
        f_X: The function value, as an array of shape (1,).
    """
    input_dim = X.shape[0]
    # The constant 418.9829 is chosen to make the minimum value 0
    constant = 418.9829 * input_dim
    summation = jnp.sum(X * jnp.sin(jnp.sqrt(jnp.abs(X))))
    result = constant - summation
    return jnp.array([result])
