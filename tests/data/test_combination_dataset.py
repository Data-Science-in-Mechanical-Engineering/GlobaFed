from jax import random
import jax.numpy as jnp

from fedbo.data import CombinationDataset
from fedbo.transformations import ObjectiveTransform, MinMaxScaling
from fedbo.visualization.data import *


if __name__ == "__main__":
    for i in range(10):
        key = random.key(i)
        num_samples = 128
        in_dim = 1
        bounds = (-5, 10)
        dataset = CombinationDataset(
            num_samples,
            in_dim,
            key,
            t=ObjectiveTransform(x_transform_post=MinMaxScaling(bounds)),
        )
        print(dataset.weights)
        # Shapes
        print(f"X shape: {dataset.X.shape}")
        print(f"f_X shape: {dataset.f_X.shape}")
        print(f"Y shape: {dataset.Y.shape}")
        assert dataset.X.shape == (num_samples, in_dim)
        assert dataset.f_X.shape == (num_samples, 1)
        assert dataset.Y.shape == (num_samples, 1)
        # Length
        print(f"Intial length: {dataset.__len__()}")
        sampel_point = jnp.full((in_dim,), bounds[1] + 2)
        dataset.add_data_point(sampel_point)
        print(f"Modified length: {dataset.__len__()}")
        # Mean and standard deviation
        print(f"Origianal mean: {dataset.Y_mean}")
        print(f"Origianal std: {dataset.Y_std}")
        print(f"Computed mean: {jnp.mean(dataset.Y)}")
        print(f"Computed std: {jnp.std(dataset.Y)}")
        assert jnp.allclose(
            jnp.mean(dataset.Y), jnp.zeros(shape=()), rtol=0.0, atol=1e-6
        )
        assert jnp.allclose(jnp.std(dataset.Y), jnp.ones(shape=()), rtol=0.0, atol=1e-6)
        print(f"Minimum value: {dataset.min_value}")
        # plot_samples(dataset)
        plot_objective(dataset)
        # plot_objective_and_samples(dataset)
