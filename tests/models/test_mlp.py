from jax import random
import jax.tree_util as jtu
import optax
from datetime import datetime

from fedbo.models import MLP
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.training.loss import MSELoss
from fedbo.visualization.model import *

if __name__ == "__main__":
    key = random.key(0)
    num_samples = 128
    in_dim = 1
    key, dataset_key = random.split(key)
    dataset = ObjectiveDataset(AckleyObjective(in_dim), num_samples, dataset_key)

    in_features = in_dim
    hidden_features = 16
    out_features = 1
    num_hidden_layers = 1
    key, model_key = random.split(key)
    model = MLP.init_from_config(in_features, hidden_features, out_features, num_hidden_layers, model_key)
    print(model.extract_config())

    num_epochs = 100
    batch_size = 64
    optimizer = optax.adamw(0.001)
    loss_func = MSELoss()
    key, train_key = random.split(key)
    # Train model
    start = datetime.now()
    new_model: MLP = model.fit(model, dataset, num_epochs, batch_size, optimizer, loss_func, train_key)
    print(f"Train model: {datetime.now() - start}")
    # Reset model
    key, reset_key = random.split(key)
    reset_model = model.reset(reset_key)
    # Train head only
    start = datetime.now()
    key, head_key = random.split(key)
    head_model: MLP = new_model.fit(
        new_model,
        dataset,
        1000,
        batch_size,
        optimizer,
        loss_func,
        head_key,
        head_only=True,
    )
    print(f"Train head: {datetime.now() - start}")

    print(f"Parameters of backbone after complete training:\n" f"{jtu.tree_leaves(new_model.backbone[0])}\n")
    print(f"Parameters backbone after head training:\n" f"{jtu.tree_leaves(head_model.backbone[0])}\n")
    print(f"Parameters of head after complete training:\n" f"{jtu.tree_leaves(new_model.head)}\n")
    print(f"Parameters head after head training:\n" f"{jtu.tree_leaves(head_model.head)}\n")

    # Plot
    visualize_prediction(jax.vmap(model), dataset)
    visualize_prediction(jax.vmap(new_model), dataset)
    visualize_prediction(jax.vmap(head_model), dataset)
    visualize_prediction(jax.vmap(reset_model), dataset)
