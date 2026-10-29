import jax
import jax.numpy as jnp
from jaxtyping import Array

from fedbo.data.objectives.base_objective import BaseObjective


class WeierstrassObjective(BaseObjective):
    """
    Weierstrass benchmark function.

    This function is continuous but nowhere differentiable, characterized
    by a fractal-like, highly rugged landscape, posing a significant
    challenge for optimization algorithms.
    """

    def __init__(self, input_dim: int):
        # Maximum value is approximated as 4 * input_dim
        super().__init__(input_dim, 0, 4 * input_dim)

    def __call__(self, X: Array) -> Array:
        """Evaluates the Weierstrass function for a given input array."""
        return weierstrass_objective(X)

    @staticmethod
    def bounds() -> tuple[float, float]:
        """Returns the recommended search space bounds for the function."""
        return (-0.5, 0.5)


@jax.jit
def weierstrass_objective(
    X: Array, a: float = 0.5, b: float = 3.0, k_max: int = 21
) -> Array:
    """
    JAX-jitted implementation of the Weierstrass function.

    The formula is:
    f(x) = sum_i(sum_k(a^k * cos(2*pi*b^k*(x_i+0.5)))) - d*sum_k(a^k*cos(pi*b^k))
    Global minimum is f(x) = 0 at x_i = 0 for all i.

    Args:
        X: Array of shape (input_dim,).
        a: A constant, typically 0.5.
        b: A constant, typically 3.
        k_max: The upper limit of the inner summation.

    Returns:
        f_X: The function value, as an array of shape (1,).
    """
    input_dim = X.shape[0]
    k = jnp.arange(k_max + 1)
    ak = jnp.power(a, k)
    bk = jnp.power(b, k)

    # First term: Sum over dimensions and k
    # Use broadcasting to compute for all x_i and k simultaneously
    # X reshaped to (d, 1), bk to (1, k_max+1)
    term1_matrix = ak * jnp.cos(2 * jnp.pi * bk * (X[:, jnp.newaxis] + 0.5))
    term1 = jnp.sum(term1_matrix)

    # Second term: Correction factor to make the minimum zero
    term2_sum = jnp.sum(ak * jnp.cos(jnp.pi * bk))
    term2 = input_dim * term2_sum

    result = term1 - term2
    return jnp.array([result])
