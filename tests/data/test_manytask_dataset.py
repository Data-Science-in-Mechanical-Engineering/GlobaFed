from jax import random
import jax.numpy as jnp

from fedbo.data import ManyTaskDataset
from fedbo.transformations import ObjectiveTransform, MinMaxScaling, GaussianNoise
from fedbo.data.objectives import (
    RosenbrockObjective,
    SchwefelObjective,
    WeierstrassObjective,
)
from fedbo.visualization.data import (
    plot_samples,
    plot_objective,
    plot_objective_and_samples,
    plot_objective_3D,
    plot_samples_3D,
)


if __name__ == "__main__":
    for _ in range(19):
        key = random.key(0)
        num_samples = 128
        in_dim = 2
        dataset = ManyTaskDataset(num_samples, in_dim, key)
        bounds = dataset.bounds
        # Shapes
        print(f"X shape: {dataset.X.shape}")
        print(f"f_X shape: {dataset.f_X.shape}")
        print(f"Y shape: {dataset.Y.shape}")
        assert dataset.X.shape == (num_samples, in_dim)
        assert dataset.f_X.shape == (num_samples, 1)
        assert dataset.Y.shape == (num_samples, 1)
        # Length
        print(f"Intial length: {dataset.__len__()}")
        data_point = jnp.full((in_dim,), bounds[1] + 2)
        dataset.add_data_point(data_point)
        print(f"Modified length: {dataset.__len__()}")
        # Mean and standard deviation
        print(f"Origianal mean: {dataset.Y_mean}")
        print(f"Origianal std: {dataset.Y_std}")
        print(f"Computed mean: {jnp.mean(dataset.Y)}")
        print(f"Computed std: {jnp.std(dataset.Y)}")
        assert jnp.allclose(
            jnp.mean(dataset.Y), jnp.zeros(shape=()), rtol=0.0, atol=1e-3
        )
        assert jnp.allclose(jnp.std(dataset.Y), jnp.ones(shape=()), rtol=0.0, atol=1e-6)

        if in_dim > 1:
            # plot_samples_3D(dataset)
            plot_objective_3D(dataset)
        else:
            # Plot
            plot_samples(dataset)
            plot_objective(dataset)
            plot_objective_and_samples(dataset)
