from jax import random
import optax

from fedbo.optimization import BayesOptWarmStart
from fedbo.training.loss import MSELoss
from fedbo.training import FederatedTrainer
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.models import MLP
from fedbo.agents import BaseAgent
from fedbo.optimization.acquisition import ThompsonSampling
from fedbo.transformations import ObjectiveTransform, MinMaxScaling
from fedbo.logger import WandBLogger

if __name__ == "__main__":
    key = random.key(0)
    num_agents = 3
    input_dim = 5

    key, train_key = random.split(key)
    num_rounds = 10
    train_cfg = {
        "num_epochs": 100,
        "batch_size": 64,
        "optimizer": optax.adamw(0.001),
        "loss_function": MSELoss(),
    }
    train_head_cfg = {
        "num_epochs": 50,
        "batch_size": 64,
        "optimizer": optax.adamw(0.001),
        "loss_function": MSELoss(),
    }
    aggr_cfg = {
        "optimizer": optax.adamw(0.01),
    }
    trainer = FederatedTrainer(num_rounds, train_cfg, train_head_cfg, aggr_cfg, key)

    dataset_cfg = {
        "objective_func": AckleyObjective(input_dim=input_dim),
        "num_samples": input_dim,
        "t": ObjectiveTransform(
            x_transform_post=MinMaxScaling(AckleyObjective.bounds()),
        ),
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
    models = [MLP.init_from_config(**model_cfg, key=subkey) for _ in range(num_agents)]
    agents = [BaseAgent(dataset, model) for dataset, model in zip(datasets, models)]

    logger = WandBLogger("federatedbo", "test", "Test_Warm_Start")

    key, subkey = random.split(key)
    optimizer = BayesOptWarmStart(
        3, agents, trainer, ThompsonSampling(), logger, 2, subkey
    )
    optimizer.run()
