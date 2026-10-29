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
    for model_cls, loss_cls in [(MLP, MSELoss), (VBLL, VBLLLoss)]:
        num_agents = 5
        input_dim = 1

        key = random.key(0)
        key, train_key = random.split(key)
        num_rounds = 10
        train_cfg = {
            "num_epochs": 10,
            "batch_size": 64,
            "optimizer": optax.adamw(0.001),
            "loss_function": loss_cls(),
            "verbose": True,
        }
        train_head_cfg = {
            "num_epochs": 50,
            "batch_size": 64,
            "optimizer": optax.adamw(0.001),
            "loss_function": loss_cls(),
        }
        aggr_cfg = {
            "optimizer": optax.adamw(0.01),
        }
        trainer = FederatedTrainer(num_rounds, train_cfg, train_head_cfg, aggr_cfg, key)

        dataset_cfg = {
            "objective_func": AckleyObjective(input_dim=input_dim),
            "num_samples": 128,
        }
        key, subkey = random.split(key)
        dataset_keys = random.split(subkey, num_agents)
        datasets = [ObjectiveDataset(**dataset_cfg, key=key) for key in dataset_keys]

        model_cfg = {
            "in_features": input_dim,
            "hidden_features": 16,
            "out_features": 1,
            "num_hidden_layers": 1,
        }
        key, subkey = random.split(key)
        models = [model_cls.init_from_config(**model_cfg, key=subkey) for _ in range(num_agents)]
        agents = [BaseAgent(dataset, model) for dataset, model in zip(datasets, models)]

        start = datetime.now()
        trained_agents = trainer(agents)
        print(f"Training time: {datetime.now() - start}")

        visualize_prediction(jax.vmap(agents[0].model), datasets[0])
        visualize_prediction(jax.vmap(trained_agents[0].model), datasets[0])

        def backbones_equal(b1, b2, tol=1e-6):
            return jtu.tree_all(jtu.tree_map(lambda x, y: jnp.allclose(x, y, atol=tol), b1, b2))

        ref_backbone = trained_agents[0].model.backbone[0]
        # Check all agents
        all_equal = all(backbones_equal(ref_backbone, agent.model.backbone[0]) for agent in trained_agents[1:])

        print("All backbones equal:", all_equal)
