from jax import random
import jax.numpy as jnp

from fedbo.data import CombinationDataset, ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.transformations import ObjectiveTransform, MinMaxScaling, RandomShifting
from fedbo.visualization.data import *


import importlib.resources

style_package = "fedbo.visualization.style"
style_file = "fedbo.mplstyle"
try:
    file_ref = importlib.resources.files(style_package).joinpath(style_file)

    with importlib.resources.as_file(file_ref) as path:
        plt.style.use(path)

except (ModuleNotFoundError, FileNotFoundError):
    print(f"Warning: Could not load the style '{style_file}' from '{style_package}'.")


def plot_objectives(datasets: list[CombinationDataset]):
    """
    Plots the objective function for a list of 1D ObjectiveDatasets.
    """
    plt.figure(figsize=(6.4, 4.8))

    dataset = datasets[0]
    assert dataset.X.shape[1] == 1, "**Visualization is only supported for 1D input!**"

    X_plot = jnp.linspace(dataset.bounds[0], dataset.bounds[1], 1000)[:, None]

    for i, dataset in enumerate(datasets):
        X, f_X, Y = dataset.objective(X_plot)
        plt.plot(X.squeeze(), f_X, linestyle="-", label=f"Agent {i}")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
    )
    plt.savefig("ackley_shifted.pdf")
    plt.show()


if __name__ == "__main__":
    datasets_to_plot = []
    for i in range(3):
        key = random.key(i)
        num_samples = 128
        in_dim = 1
        bounds = (-5, 10)
        key, a, b = random.split(key, 3)
        dataset = ObjectiveDataset(
            AckleyObjective(1),
            num_samples,
            key,
            t=ObjectiveTransform(
                x_transform_pre=RandomShifting((-8, 3), a),
                f_x_transform=RandomShifting((-10, 10), b),
            ),
        )
        datasets_to_plot.append(dataset)

    plot_objectives(datasets_to_plot)
