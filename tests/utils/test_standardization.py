from jax import random
import jax.numpy as jnp

from fedbo.transformations import standardize, unstandardize
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import ackley_objective


def compute_mean_and_std(X):
    mean_X = jnp.mean(X)
    std_X = jnp.std(X)
    print(f"Mean: {mean_X}, Std: {std_X}")
    return mean_X, std_X


if __name__ == "__main__":
    key = random.key(0)
    num_samples = 128
    in_dim = 1
    bounds = (-5, 10)
    dataset = ObjectiveDataset(
        ackley_objective,
        num_samples,
        bounds,
        in_dim,
        0,
        key,
    )

    pre_mean, pre_std = compute_mean_and_std(dataset.f_X)

    stand_f_x = standardize(dataset.f_X, pre_mean, pre_std)
    mean_fx, std_fx = compute_mean_and_std(stand_f_x)
    assert jnp.allclose(mean_fx, jnp.zeros(shape=()), rtol=0.0, atol=1e-6)
    assert jnp.allclose(std_fx, jnp.ones(shape=()), rtol=0.0, atol=1e-6)

    unstand_f_x = unstandardize(stand_f_x, pre_mean, pre_std)
    mean_un_fx, std_un_fx = compute_mean_and_std(unstand_f_x)
    assert jnp.allclose(mean_un_fx, pre_mean, rtol=0.0, atol=1e-6)
    assert jnp.allclose(std_un_fx, pre_std, rtol=0.0, atol=1e-6)
