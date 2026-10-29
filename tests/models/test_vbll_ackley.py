from jax import random
import jax.tree_util as jtu
import optax
from datetime import datetime

from fedbo.models.vbll import VBLL
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.training.loss import VBLLLoss
from fedbo.visualization.model import *

if __name__ == "__main__":
    key = random.key(5)
    num_samples = 128
    in_dim = 1
    key, dataset_key = random.split(key)
    dataset = ObjectiveDataset(AckleyObjective(in_dim), num_samples, dataset_key)

    in_features = in_dim
    hidden_features = 64
    out_features = 1
    num_hidden_layers = 4
    key, model_key = random.split(key)
    model = VBLL.init_from_config(in_features, hidden_features, out_features, num_hidden_layers, model_key, reg_factor=1.0)
    print(model.extract_config())

    num_epochs = 1000
    batch_size = 32
    optimizer = optax.adamw(1e-3, weight_decay=0.0)
    loss_func = VBLLLoss()
    key, train_key = random.split(key)
    # Train model
    start = datetime.now()
    new_model = model.fit(model, dataset, num_epochs, batch_size, optimizer, loss_func, train_key, verbose=True)
    print(f"Train model: {datetime.now() - start}")
    # Reset model
    key, reset_key = random.split(key)
    reset_model = model.reset(reset_key)
    # Train head only
    start = datetime.now()
    key, head_key = random.split(key)
    head_model = new_model.fit(
        new_model, dataset, 100, batch_size, optimizer, loss_func, head_key, head_only=True, verbose=True
    )
    print(f"Train head: {datetime.now() - start}")

    # Plot
    vizualize_vbll(model, dataset)
    vizualize_vbll(new_model, dataset)
    vizualize_vbll(head_model, dataset)
    vizualize_vbll(reset_model, dataset)
