from jax import random
import matplotlib.pyplot as plt
from datetime import datetime

from fedbo.optimization.acquisition import FTSAcquisitionFunc
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.models import FTSGaussianProcess, PredictionModel
from fedbo.transformations import MinMaxScaling, ObjectiveTransform
from fedbo.agents import BaseAgent

from fedbo.visualization.model import *


if __name__ == "__main__":
    key = random.key(1)
    # Dataset
    num_samples = 4
    in_dim = 1
    key, objective_key = random.split(key)
    dataset = ObjectiveDataset(
        AckleyObjective(in_dim),
        num_samples,
        objective_key,
        t=ObjectiveTransform(x_transform_post=MinMaxScaling(AckleyObjective.bounds())),
    )
    key, model_key = random.split(key)
    model = FTSGaussianProcess(model_key)
    # Training
    model = model.fit(model, dataset)
    # Acquision
    start = datetime.now()
    sampler = FTSAcquisitionFunc(1, in_dim)
    key, subkey = random.split(key)
    x_next, y_next = sampler.normal_acquisition(model, dataset, in_dim, subkey)
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

    num_agents = 5
    dataset_cfg = {
        "objective_func": AckleyObjective(input_dim=in_dim),
        "num_samples": 128,
        "t": ObjectiveTransform(
            x_transform_post=MinMaxScaling(AckleyObjective.bounds())
        ),
    }
    key, subkey = random.split(key)
    dataset_keys = random.split(subkey, num_agents)
    datasets = [ObjectiveDataset(**dataset_cfg, key=key) for key in dataset_keys]

    key, subkey = random.split(key)
    model_keys = random.split(subkey, num_agents)
    models = [FTSGaussianProcess(key) for key in model_keys]
    agents = [BaseAgent(dataset, model) for dataset, model in zip(datasets, models)]
    # Federated Acquision
    start = datetime.now()
    key, subkey = random.split(key)
    x_next, y_next = sampler.federated_acquisition(agents, in_dim, subkey)
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
