from jax import random
import jax.tree_util as jtu
import optax
from datetime import datetime

from fedbo.training import CentralizedTrainer
from fedbo.training.loss import MSELoss, VBLLLoss
from fedbo.agents import BaseAgent
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.models import MLP, VBLL
from fedbo.visualization.model import *


if __name__ == "__main__":
    for model_type, loss_type in zip([MLP, VBLL], [MSELoss, VBLLLoss]):

        num_agents = 5
        input_dim = 1

        key = random.key(0)
        key, train_key = random.split(key)
        num_rounds = 50
        l_train_cfg = {
            "num_epochs": 25,
            "batch_size": 64,
            "optimizer": optax.adamw(1e-4),
            "loss_function": loss_type(),
        }
        g_train_cfg = {
            "num_epochs": 10,
            "batch_size": 64,
            "optimizer": optax.adamw(8e-4),
            "loss_function": loss_type(),
        }
        h_train_cfg = {
            "num_epochs": 25,
            "batch_size": 64,
            "optimizer": optax.adamw(1e-4),
            "loss_function": loss_type(),
        }
        trainer = CentralizedTrainer(num_rounds, l_train_cfg, g_train_cfg, h_train_cfg, key)

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
        models = [model_type.init_from_config(**model_cfg, key=subkey) for _ in range(num_agents)]
        agents = [BaseAgent(dataset, model) for dataset, model in zip(datasets, models)]

        start = datetime.now()
        trained_agents = trainer(agents)
        print(f"Training time: {datetime.now() - start}")

        visualize_prediction(jax.vmap(agents[0].model), datasets[0])
        visualize_prediction(jax.vmap(trained_agents[0].model), datasets[0])

        # Check all agents
        def backbones_equal(b1, b2, tol=1e-6):
            return jtu.tree_all(jtu.tree_map(lambda x, y: jnp.allclose(x, y, atol=tol), b1, b2))

        ref_backbone = trained_agents[0].model.backbone[0]
        all_equal = all(backbones_equal(ref_backbone, agent.model.backbone[0]) for agent in trained_agents[1:])
        print("All backbones equal:", all_equal)

        # for agent in trained_agents:
        #     print(f"Parameters of backbone after complete training:\n" f"{jtu.tree_leaves(agent.model.backbone[0])}\n")
