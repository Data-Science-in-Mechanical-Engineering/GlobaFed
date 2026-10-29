from jax import random
import jax.numpy as jnp

from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import ackley_objective
from fedbo.transformations import (
    RandomShifting,
    MinMaxScaling,
    GaussianNoise,
    RandomScaling,
    ObjectiveTransform,
)
from fedbo.visualization.data import (
    plot_samples,
    plot_objective,
    plot_transformations,
    plot_objective_and_samples,
    plot_samples_3D,
    plot_objective_3D,
)


if __name__ == "__main__":
    key = random.key(0)

    key, subkey = random.split(key)
    gaussian_noise = GaussianNoise(0.05, subkey)
    key, subkey = random.split(key)
    shifting = RandomShifting((-2, 2), subkey)
    key, subkey = random.split(key)
    scaling = RandomScaling((0.1, 2), subkey)
    key, subkey = random.split(key)
    minmax = MinMaxScaling((-5, 10), subkey)

    dataset = ObjectiveDataset(
        ackley_objective,
        128,
        (-5, 10),
        1,
        0,
        key,
        ObjectiveTransform(shifting, minmax, scaling, gaussian_noise),
    )
    print(f"X shape: {dataset.X.shape}")
    print(f"Y: {dataset.Y.shape}")
    print(dataset.min_value)
    plot_samples(dataset)
    plot_objective(dataset)
    plot_transformations(dataset)
    print(f"Intial length: {dataset.__len__()}")
    dataset.add_sample_point(jnp.array([12]))
    print(f"Modified length: {dataset.__len__()}")
    plot_samples(dataset)
    plot_objective(dataset)
    plot_objective_and_samples(dataset)

    # 3D
    key, subkey = random.split(key)
    gaussian_noise = GaussianNoise(1, subkey)
    key, subkey = random.split(key)
    shifting = RandomShifting((-20, 20), subkey)
    key, subkey = random.split(key)
    scaling = RandomScaling((0.1, 2), subkey)

    dataset = ObjectiveDataset(
        ackley_objective,
        128,
        (-5, 10),
        2,
        0,
        key,
        ObjectiveTransform(shifting, minmax, scaling, gaussian_noise),
    )
    print(f"X shape: {dataset.X.shape}")
    print(f"Y: {dataset.Y.shape}")
    plot_samples_3D(dataset)
    plot_objective_3D(dataset)
