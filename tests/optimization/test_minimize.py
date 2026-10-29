import jax
from jax import random
import jax.numpy as jnp
import equinox as eqx
import optimistix

from fedbo.config import CosineDatasetConfig, FivePeaksDatasetConfig
from fedbo.data import CosineDataset, FivePeaksDataset


def example():
    # Define your objective function
    def rosenbrock(x, args):
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    # Initial guess
    x0 = jnp.array([1.2, 1.2])

    # Specify tolerances (you can adjust these)
    rtol = 1e-5
    atol = 1e-8

    # Define the solver
    solver = optimistix.BFGS(rtol, atol)

    # Minimize the function
    result = optimistix.minimise(rosenbrock, solver, x0)

    print("Minimizer:", result.value)
    print("Stats:", result.stats)


def example2():
    key = random.key(0)

    key, subkey = random.split(key)
    data_cfg = CosineDatasetConfig(128, 0.0, (-6, 6))
    dataset = CosineDataset(data_cfg, key)
    dataset.visualize_objective()

    def func(x, args):
        print(args)
        return -dataset.objective(x, 0.0)

    # Initial guess
    key, subkey = random.split(key)
    x0 = jax.random.uniform(subkey)
    print(x0)

    # Specify tolerances (you can adjust these)
    rtol = 1e-5
    atol = 1e-8

    # Define the solver
    solver = optimistix.BFGS(rtol, atol)

    # Minimize the function
    result = optimistix.minimise(func, solver, x0)

    print("Minimizer:", result.value)
    print("Stats:", result.stats)


def example3(solver):
    key = random.key(0)

    key, subkey = random.split(key)
    data_cfg = CosineDatasetConfig(128, 0.0, (-6, 6))
    dataset = CosineDataset(data_cfg, key)
    # dataset.visualize_objective()

    def func(x, args):
        return -dataset.objective(x, 0.0)

    # Initial guess
    key, subkey = random.split(key)
    x0 = jax.random.uniform(subkey)
    print(x0)

    # Minimize the function
    result = optimistix.minimise(func, solver, x0)

    print("Minimizer:", result.value)
    print("Stats:", result.stats)


if __name__ == "__main__":
    # Specify tolerances (you can adjust these)
    rtol = 1e-5
    atol = 1e-8

    # Define the solver
    solvers = [
        optimistix.BFGS(rtol, atol),
        optimistix.NonlinearCG(rtol, atol),
        optimistix.BestSoFarMinimiser(optimistix.BFGS(rtol, atol)),
        optimistix.BestSoFarMinimiser(optimistix.NonlinearCG(rtol, atol)),
    ]

    for solver in solvers:
        example3(solver)
