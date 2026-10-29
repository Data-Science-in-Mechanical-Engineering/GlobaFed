import jax
from jax import random
import optax
from datetime import datetime

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from fedbo.training import LocalTrainer
from fedbo.training.loss import MSELoss
from fedbo.agents import BaseAgent
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.models import MLP, GaussianProcess
from fedbo.visualization.model import *


if __name__ == "__main__":
    num_agents = 5
    input_dim = 1

    key = random.key(0)
    key, train_key = random.split(key)
    train_cfg = {
        "num_epochs": 1000,
        "batch_size": 64,
        "optimizer": optax.adamw(0.001),
        "loss_function": MSELoss(),
    }

    trainer = LocalTrainer(train_cfg, key)

    dataset_cfg = {
        "objective_func": AckleyObjective(input_dim=input_dim),
        "num_samples": 128,
    }
    key, subkey = random.split(key)
    dataset_keys = random.split(subkey, num_agents)
    datasets = [ObjectiveDataset(**dataset_cfg, key=key) for key in dataset_keys]

    # MLP
    model_cfg = {
        "in_features": input_dim,
        "hidden_features": 16,
        "out_features": 1,
        "num_hidden_layers": 1,
    }
    key, subkey = random.split(key)
    key, subkey = random.split(key)
    models = [MLP.init_from_config(**model_cfg, key=subkey) for _ in range(num_agents)]
    agents = [BaseAgent(dataset, model) for dataset, model in zip(datasets, models)]

    start = datetime.now()
    trained_agents = trainer(agents)
    print(f"Training time: {datetime.now() - start}")

    visualize_prediction(jax.vmap(agents[0].model), datasets[0])
    visualize_prediction(jax.vmap(trained_agents[0].model), datasets[0])

    # GP
    trainer = LocalTrainer({}, key)
    key, subkey = random.split(key)
    model_keys = random.split(subkey, num_agents)
    models = [GaussianProcess(key=key) for key in model_keys]
    agents = [BaseAgent(dataset, model) for dataset, model in zip(datasets, models)]

    start = datetime.now()
    trained_agents = trainer(agents)
    print(f"Training time: {datetime.now() - start}")

    # Note: gp.fit has side effects
    visualize_prediction(agents[0].model, datasets[0])
    visualize_prediction(trained_agents[0].model, datasets[0])
