from jax import random
import jax.tree_util as jtu
import jax.numpy as jnp
import optax
from datetime import datetime
import jax

from fedbo.training import FederatedTrainer
from fedbo.training.loss import MSELoss, VBLLLoss
from fedbo.agents import BaseAgent
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.models import MLP, VBLL
from fedbo.visualization.model import *


if __name__ == "__main__":
    input_dim = 1

    key = random.key(0)
    key, train_key = random.split(key)
    train_cfg = {
        "num_epochs": 10,
        "batch_size": 64,
        "optimizer": optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(0.001)),
        "loss_function": MSELoss(),
        "verbose": False,
    }

    dataset_cfg = {
        "objective_func": AckleyObjective(input_dim=input_dim),
        "num_samples": 128,
    }
    key, subkey = random.split(key)
    dataset = ObjectiveDataset(**dataset_cfg, key=subkey)

    model_cfg = {
        "in_features": input_dim,
        "hidden_features": 16,
        "out_features": 1,
        "num_hidden_layers": 1,
    }
    key, subkey = random.split(key)
    print("MLP")
    model = MLP.init_from_config(**model_cfg, key=subkey)
    model = MLP.fit(model, dataset, **train_cfg, key=subkey)
    model = MLP.init_from_config(**model_cfg, key=key)
    model = MLP.fit(model, dataset, **train_cfg, key=key)

    print("VBLL")
    train_cfg = {
        "num_epochs": 10,
        "batch_size": 64,
        "optimizer": optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(0.001)),
        "loss_function": VBLLLoss(),
        "verbose": False,
    }
    model = VBLL.init_from_config(**model_cfg, key=subkey)
    model = VBLL.fit(model, dataset, **train_cfg, key=subkey)
    model_cfg = {
        "in_features": input_dim,
        "hidden_features": 16,
        "out_features": 1,
        "num_hidden_layers": 1,
        "reg_factor": 1,
    }
    model = VBLL.init_from_config(**model_cfg, key=subkey)
    model = VBLL.fit(model, dataset, **train_cfg, key=subkey)
    key, a = random.split(key)
    model = VBLL.init_from_config(**model_cfg, key=a)
    model = VBLL.fit(model, dataset, **train_cfg, key=subkey)
