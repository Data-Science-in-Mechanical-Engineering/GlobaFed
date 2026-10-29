from jax import random
import optax
import matplotlib.pyplot as plt
from datetime import datetime

from fedbo.optimization.acquisition import ThompsonSampling
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.models import MLP, PredictionModel
from fedbo.training.loss import MSELoss
from fedbo.transformations import MinMaxScaling, ObjectiveTransform

from fedbo.visualization.model import *


if __name__ == "__main__":
    key = random.key(1)
    # Dataset
    num_samples = 8
    in_dim = 1
    bounds = (-5, 10)
    key, objective_key, t_key = random.split(key, 3)
    dataset = ObjectiveDataset(
        AckleyObjective(in_dim),
        num_samples,
        objective_key,
        t=ObjectiveTransform(x_transform_post=MinMaxScaling(bounds, t_key)),
    )
    # Model
    in_features = in_dim
    hidden_features = 64
    out_features = 1
    num_hidden_layers = 3
    key, model_key = random.split(key)
    model = MLP.init_from_config(in_features, hidden_features, out_features, num_hidden_layers, model_key)
    # Training
    num_epochs = 200
    batch_size = 64
    optimizer = optax.adamw(0.001)
    loss_func = MSELoss()
    key, train_key = random.split(key)
    model: MLP = model.fit(model, dataset, num_epochs, batch_size, optimizer, loss_func, train_key)
    # Acquision
    start = datetime.now()
    sampler = ThompsonSampling((0, 1))
    key, subkey = random.split(key)
    x_next, y_next = sampler(model, in_dim, subkey)
    print(f"Time: {datetime.now() - start}")
    print(f"Sampled point: ({x_next}, {y_next})")
    dataset.add_sampled_point(x_next)
    print(f"X[-1]: {dataset.X[-1]}")
    print(f"f_X[-1]: {dataset.f_X[-1]}")
    print(f"Y[-1]: {dataset.Y[-1]}")
    print(f"Undstandardized Y[-1]: {dataset.unstandardized_Y[-1]}")
    fig = create_model_prediction_fig(
        PredictionModel(model, dataset.unstandardize_func),
        dataset,
        dataset.X[-1],
        dataset.unstandardized_Y[-1],
    )
    plt.show()
