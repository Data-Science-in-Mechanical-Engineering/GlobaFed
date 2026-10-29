from jax import random
import optax
import matplotlib.pyplot as plt
from datetime import datetime

from fedbo.optimization.acquisition import ThompsonSampling
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.models import VBLL, VBLLSampleModel
from fedbo.utils.minimize import minimize_func
from fedbo.training.loss import VBLLLoss
from fedbo.transformations import MinMaxScaling, ObjectiveTransform

from fedbo.visualization.model import *


def visualize_vbll_thompson_sampling(
    model: VBLL, dataset: ObjectiveDataset, key, x_next, y_next, stdevs=1.0, num_samples=100
):
    X = jnp.linspace(0, 1, 1000)[:, None]
    Xp = X.squeeze()

    predictions = jax.vmap(model.predictive)(X)
    Y_mean = predictions.loc.squeeze()
    Y_stdev = jnp.sqrt(predictions.covariance.squeeze())

    # Thompson Sampling
    keys = random.split(key, num_samples)
    for k in keys:
        sample_key, optim_key = random.split(k)
        sample_model = VBLLSampleModel(model, sample_key)
        ts_pred = jax.vmap(sample_model)(X)
        plt.plot(X, ts_pred, color="orange", label="Thompson Sample")

    plt.plot(X, Y_mean, label="Mean Prediction")
    plt.fill_between(Xp, Y_mean - stdevs * Y_stdev, Y_mean + stdevs * Y_stdev, alpha=0.2, color="b")
    plt.fill_between(Xp, Y_mean - 2 * stdevs * Y_stdev, Y_mean + 2 * stdevs * Y_stdev, alpha=0.2, color="b")
    plt.scatter(dataset.X, dataset.Y, color="k", label="Ground Truth")
    plt.scatter(x_next, y_next, color="r", marker="x", label="Samples")
    plt.show()


if __name__ == "__main__":
    key = random.key(5)
    # Dataset
    num_samples = 4
    in_dim = 1
    key, objective_key, t_key = random.split(key, 3)
    dataset = ObjectiveDataset(
        AckleyObjective(in_dim),
        num_samples,
        objective_key,
        t=ObjectiveTransform(x_transform_post=MinMaxScaling(AckleyObjective.bounds(), t_key)),
    )
    # Model
    in_features = in_dim
    hidden_features = 64
    out_features = 1
    num_hidden_layers = 4
    key, model_key = random.split(key)
    model = VBLL.init_from_config(in_features, hidden_features, out_features, num_hidden_layers, model_key, reg_factor=1.0)
    # Training
    num_epochs = 1000
    batch_size = 64
    optimizer = optax.adamw(0.001, weight_decay=0.0)
    loss_func = VBLLLoss()
    key, train_key = random.split(key)
    model = model.fit(model, dataset, num_epochs, batch_size, optimizer, loss_func, train_key, verbose=True)
    # Acquision
    start = datetime.now()
    sampler = ThompsonSampling((0, 1))
    key, subkey = random.split(key)
    x_next, y_next = sampler(model, in_dim, subkey)
    print(f"Sample Time: {datetime.now() - start}")
    print(f"Sampled point: ({x_next}, {y_next})")
    visualize_vbll_thompson_sampling(model, dataset, subkey, x_next, y_next)
