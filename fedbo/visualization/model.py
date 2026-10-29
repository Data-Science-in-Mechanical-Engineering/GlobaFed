from __future__ import annotations
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from jaxtyping import Array
    from fedbo.models import PredictionModel, BaseModel, VBLL
    from fedbo.models.vbll import Normal
    from fedbo.data import ObjectiveDataset


def create_model_prediction_fig(
    model: PredictionModel,
    dataset: ObjectiveDataset,
    X_Samples: Array = None,
    Y_Samples: Array = None,
):
    X = jnp.linspace(dataset.bounds[0], dataset.bounds[1], 1000)[:, None]
    X, f_X, Y = dataset.objective(X)
    Y_pred = jax.vmap(model)(X) if model.can_vmap else model(X)
    fig, ax = plt.subplots()
    ax.plot(X, Y, label="Objective Y")
    ax.plot(X, f_X, label=" Objective f(X)")
    ax.plot(X, Y_pred, label="Prediction")
    # Unstandardize Y
    ax.scatter(
        dataset.X,
        dataset.unstandardized_Y,
        color="k",
        label="Ground Truth",
    )
    if not (X_Samples is None or Y_Samples is None):
        ax.scatter(X_Samples, Y_Samples, color="r", marker="x", label="Samples")
    return fig


def visualize_prediction(model: BaseModel, dataset: ObjectiveDataset):
    X = jnp.linspace(dataset.bounds[0], dataset.bounds[1], 1000)[:, None]
    Y_pred = jax.vmap(model(X)) if isinstance(model, eqx.Module) else model(X)
    plt.plot(X, Y_pred, label="Prediction")
    plt.scatter(dataset.X, dataset.Y, color="red", marker="x", label="Ground Truth")
    plt.title("Model prediction")
    plt.show()


def visualize_prediction_with_samples(
    model: PredictionModel,
    dataset: ObjectiveDataset,
    X_Samples: Array = None,
    Y_Samples: Array = None,
):
    fig = create_model_prediction_fig(model, dataset, X_Samples, Y_Samples)
    plt.show()
    plt.close(fig)


def vizualize_vbll(model: VBLL, dataset: ObjectiveDataset, stdevs=1.0):
    """Visualize VBLL model predictions, including predictive uncertainty."""
    X = jnp.linspace(dataset.bounds[0], dataset.bounds[1], 1000)[:, None]
    Xp = X.squeeze()

    predictions = jax.vmap(model.predictive)(X)
    Y_mean = predictions.loc.squeeze()
    Y_stdev = jnp.sqrt(predictions.covariance.squeeze())

    plt.plot(Xp, Y_mean)
    plt.fill_between(Xp, Y_mean - stdevs * Y_stdev, Y_mean + stdevs * Y_stdev, alpha=0.2, color="b")
    plt.fill_between(Xp, Y_mean - 2 * stdevs * Y_stdev, Y_mean + 2 * stdevs * Y_stdev, alpha=0.2, color="b")
    plt.scatter(dataset.X, dataset.Y, color="k")
    plt.show()
