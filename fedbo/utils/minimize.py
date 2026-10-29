import jax
from jax import random
import jax.numpy as jnp
from jaxopt import LBFGSB
import numpy as np
from scipy import optimize
from scipy.stats.qmc import Sobol
from typing import Callable
from jaxtyping import Array, Key


def minimize_func(
    func: Callable[[Array], Array],
    input_dim: int,
    bounds: tuple[int, int],
    key: Key,
    num_restarts: int = 10,
    max_iter: int = 100,
) -> tuple[Array, Array]:
    """
    Minimize a function `func` over a bounded domain using multiple restarts with L-BFGS-B.

    Args:
        func: Function to minimize. Input shape (input_dim,), output scalar or (1,).
        input_dim: Dimensionality of input space.
        bounds: Tuple (lower_bound, upper_bound) defining the box constraints.
        key: JAX PRNGKey for random number generation.
        num_restarts: Number of random restarts.
        max_iter: Maximum number of iterations for the L-BFGS-B algorithm

    Returns:
        best_x: The input vector with the lowest function value found.
        best_y: The lowest function value.
    """
    lb, ub = bounds
    lbfgsb = LBFGSB(fun=lambda x: func(x).squeeze(), maxiter=max_iter)

    # Generate multiple random initial points
    keys = random.split(key, num_restarts)
    x0s = jnp.array([lb + (ub - lb) * random.uniform(k, (input_dim,)) for k in keys])

    # JIT compile a single run of the optimizer
    @jax.jit
    def single_run(x0):
        result = lbfgsb.run(x0, bounds=bounds)
        y = func(result.params)
        return result.params, y

    # Vectorize over the batch of initial points
    params_batch, y_batch = jax.vmap(single_run)(x0s)

    # Filter out entries with nan
    y_batch = jnp.where(jnp.isnan(y_batch), jnp.inf, y_batch)

    # Find best result
    best_idx = jnp.argmin(y_batch)
    best_x = params_batch[best_idx]
    best_y = y_batch[best_idx]

    # Clip params within bounds just in case
    best_x = jnp.clip(best_x, lb, ub)
    return best_x, best_y


# Note: Consumes a lot of memory with repeated use
def minimize_func_scan(
    func: Callable[[Array], Array],
    input_dim: int,
    bounds: tuple[int, int],
    key: Key,
    num_restarts: int = 10,
    max_iter: int = 100,
):
    lb, ub = bounds
    lbfgsb = LBFGSB(fun=lambda x: func(x).squeeze(), maxiter=max_iter)

    def body(carry, _):
        key = carry
        key, subkey = random.split(key)
        x0 = lb + (ub - lb) * random.uniform(subkey, (input_dim,))
        result = lbfgsb.run(x0, bounds=bounds)
        y = func(result.params)
        return key, (result.params, y)

    key, results = jax.lax.scan(body, key, length=num_restarts)

    X_Cand, Y_Cand = results
    # Clip to bounds for numerical safety
    X_Cand = jnp.clip(X_Cand, lb, ub)
    best_idx = jnp.argmin(Y_Cand)
    best_x = X_Cand[best_idx]
    best_y = Y_Cand[best_idx]
    return best_x, best_y


def approximate_minimum(
    func: Callable[[Array], Array],
    input_dim: int,
    bounds: tuple[int, int],
    key: Key,
    num_evals: int = 10000,
) -> float:
    """
    Approximates the global minimum of a function by evaluating it
    at `num_evals` random points within the given bounds.

    Args:
        func: Function to minimize. Should take a JAX array of shape (..., input_dim)
              and return a 1D array of function values.
        input_dim: Dimensionality of input space.
        bounds: Tuple (lb, ub) defining the search space.
        num_evals: Number of random samples to evaluate.
        key: JAX PRNGKey for reproducibility.

    Returns:
        Approximate minimum value of the function over the domain.
    """
    np_key = np.random.default_rng(np.asarray(random.key_data(key)))
    # Draw samples
    sobol_engine = Sobol(d=input_dim, scramble=True, rng=np_key)
    sobol_samples = sobol_engine.random_base2(m=int(np.log2(num_evals)))
    # Scale to bounds
    lb, ub = bounds
    samples = jnp.array(lb + sobol_samples * (ub - lb))
    # Evaluate the function on all samples
    values = jax.vmap(func)(samples)
    min_idx = jnp.argmin(values)
    # Return the minimum value found
    return samples[min_idx], jnp.min(values)


def minimize_func_unoptimized(
    func: Callable[[Array], Array],
    input_dim: int,
    bounds: tuple[int, int],
    key: Key,
    num_restarts=10,
):
    def func_wrapper(x: Array):
        return func(x).squeeze()

    lb, ub = bounds
    lbfgsb = LBFGSB(fun=func_wrapper, maxiter=50)

    X_Cand = jnp.empty((num_restarts, input_dim))
    Y_Cand = jnp.empty((num_restarts, 1))
    keys = random.split(key, num_restarts)
    for j in range(num_restarts):
        x0 = lb + (ub - lb) * jax.random.uniform(keys[j], shape=(input_dim,))

        result = lbfgsb.run(x0, bounds=bounds)

        X_Cand = X_Cand.at[j, :].set(result.params)
        Y_Cand = Y_Cand.at[j].set(func(result.params))

    X_Cand = jnp.clip(X_Cand, lb, ub)
    best_idx = jnp.argmin(Y_Cand)
    best_x = X_Cand[best_idx]
    best_y = Y_Cand[best_idx]
    return best_x, best_y


def minimize_func_scipy(
    func: Callable[[Array], Array],
    input_dim: int,
    bounds: tuple[int, int],
    key: Key,
    num_restarts: int = 10,
):
    lb, ub = bounds

    X_Cand = jnp.empty((num_restarts, input_dim))
    Y_Cand = jnp.empty((num_restarts, 1))
    keys = random.split(key, num_restarts)

    def f(x):
        return func(x).squeeze()

    def grad_func(x):
        # Mean to get a scalar
        return jax.grad(f)(x)

    for j in range(num_restarts):
        x0 = lb + (ub - lb) * jax.random.uniform(keys[j], shape=(input_dim,))

        res = optimize.minimize(f, x0, jac=grad_func, bounds=[(lb, ub)], method="L-BFGS-B")

        X_Cand = X_Cand.at[j, :].set(res.x)
        Y_Cand = Y_Cand.at[j].set(res.fun)

    X_Cand = jnp.clip(X_Cand, lb, ub)
    best_idx = jnp.argmin(Y_Cand)
    best_x = X_Cand[best_idx]
    best_y = Y_Cand[best_idx]
    return best_x, best_y
