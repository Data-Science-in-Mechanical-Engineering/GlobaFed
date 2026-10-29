import jax
from jax import random
import optax
from datetime import datetime

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from fedbo.training import LocalTrainer
from fedbo.agents import BaseAgent
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.utils.fts.fts_gaussian_process import FTSGaussianProcess
from fedbo.visualization.model import *
from fedbo.transformations import ObjectiveTransform, MinMaxScaling


if __name__ == "__main__":
    num_agents = 5
    input_dim = 1

    key = random.key(0)
    key, t_key = random.split(key)

    dataset_cfg = {
        "objective_func": AckleyObjective(input_dim=input_dim),
        "num_samples": 128,
        "t": ObjectiveTransform(
            x_transform_post=MinMaxScaling(AckleyObjective.bounds(), t_key)
        ),
    }
    key, subkey = random.split(key)
    dataset_keys = random.split(subkey, num_agents)
    datasets = [ObjectiveDataset(**dataset_cfg, key=key) for key in dataset_keys]

    trainer = LocalTrainer({}, key)
    key, subkey = random.split(key)
    model_keys = random.split(subkey, num_agents)
    models = [FTSGaussianProcess(key=key) for key in model_keys]
    agents = [BaseAgent(dataset, model) for dataset, model in zip(datasets, models)]

    start = datetime.now()
    trained_agents = trainer(agents)
    print(f"Training time: {datetime.now() - start}")

    # Note: gp.fit has side effects
    # visualize_prediction(agents[0].model, datasets[0])
    visualize_prediction_with_samples(trained_agents[0].pred_model, datasets[0])
