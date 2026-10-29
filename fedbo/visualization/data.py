from __future__ import annotations
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fedbo.data import ObjectiveDataset


def plot_samples(dataset: ObjectiveDataset):
    assert dataset.X.shape[1] == 1, "Visualization is only supported for 1D input!"
    plt.scatter(dataset.X, dataset.Y)
    plt.title("Samples")
    plt.show()


def plot_objective(dataset: ObjectiveDataset):
    assert dataset.X.shape[1] == 1, "Visualization is only supported for 1D input!"
    X = jnp.linspace(dataset.bounds[0], dataset.bounds[1], 1000)[:, None]
    X, f_X, Y = dataset.objective(X)
    plt.title("Objective Function")
    plt.plot(X, Y, label="Y")
    plt.plot(X, f_X, label="f(X)")
    plt.legend()
    plt.show()


def plot_transformations(dataset: ObjectiveDataset):
    assert dataset.X.shape[1] == 1, "Visualization is only supported for 1D input!"
    X_ORIGINAL = jnp.linspace(dataset.bounds[0], dataset.bounds[1], 1000)[:, None]
    f_X_ORIGINAL = dataset._objective(X_ORIGINAL)
    X, f_X, Y = dataset.objective(X_ORIGINAL)
    plt.title("Transformations")
    plt.plot(X, Y, label="Y Transformed")
    plt.plot(X, f_X, label="f(X) Transformed")
    plt.plot(X_ORIGINAL, f_X_ORIGINAL, label="f(x) Original")
    plt.legend()
    plt.show()


def plot_objective_and_samples(dataset: ObjectiveDataset):
    assert dataset.X.shape[1] == 1, "Visualization is only supported for 1D input!"
    X_obj = jnp.linspace(dataset.bounds[0], dataset.bounds[1], 1000)[:, None]
    X_obj, f_X_obj, Y_obj = dataset.objective(X_obj)
    plt.title("Objective Function and Samples")
    plt.scatter(dataset.X, dataset.Y, color="red", marker="x", label="Samples")
    plt.plot(X_obj, Y_obj, label="Y")
    plt.plot(X_obj, f_X_obj, label="f(X)")
    plt.legend()
    plt.show()


def plot_samples_3D(dataset: ObjectiveDataset):
    assert dataset.X.shape[1] == 2, "Visualization is only supported for 2D input!"
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x = dataset.X[:, 0]
    y = dataset.X[:, 1]
    z = dataset.Y[:, 0]

    ax.scatter(x, y, z)
    ax.set_title("Samples")
    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.set_zlabel("Y")
    plt.show()


def plot_objective_3D(dataset: ObjectiveDataset):
    assert dataset.X.shape[1] == 2, "Visualization is only supported for 2D input!"

    # Create a grid of x, y values
    x_vals = jnp.linspace(dataset.bounds[0], dataset.bounds[1], 100)
    y_vals = jnp.linspace(dataset.bounds[0], dataset.bounds[1], 100)
    X_grid, Y_grid = jnp.meshgrid(x_vals, y_vals)

    # Flatten the grid for evaluation
    X_flat = jnp.vstack((X_grid.ravel(), Y_grid.ravel())).T

    # Evaluate the objective function
    _, _, Z_flat = dataset.objective(X_flat)

    # Reshape Z to match the shape of the grid
    Z_grid = Z_flat.reshape(X_grid.shape)

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Plot the surface
    ax.plot_surface(X_grid, Y_grid, Z_grid, cmap="viridis")

    ax.set_title("Objective Function Surface")
    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.set_zlabel("Y")

    plt.show()
