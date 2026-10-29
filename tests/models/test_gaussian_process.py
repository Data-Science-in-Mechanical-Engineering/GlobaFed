import jax

jax.config.update("jax_enable_x64", True)
from jax import random

from fedbo.agents import BaseAgent
from fedbo.models import GaussianProcess
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.optimization.acquisition import ThompsonSampling, ExpectedImprovement
from fedbo.visualization.model import visualize_prediction_with_samples
from fedbo.transformations import (
    ObjectiveTransform,
    RandomShifting,
    MinMaxScaling,
    GaussianNoise,
)
from datetime import datetime

if __name__ == "__main__":
    start = datetime.now()

    key = random.key(0)
    key, subkey = random.split(key)
    model = GaussianProcess(subkey)
    keys = random.split(key, 5)
    t = ObjectiveTransform(
        RandomShifting((-0.3, 0.3), keys[1]),
        MinMaxScaling((-5, 10), keys[2]),
        RandomShifting((1, 2), keys[3]),
        GaussianNoise(0.3, keys[4]),
    )
    key = keys[0]
    key, subkey = random.split(key)
    dataset = ObjectiveDataset(AckleyObjective(1), 8, subkey, t)
    print(t.f_x_transform.shift_factor)
    print(dataset.min_value)
    agent = BaseAgent(dataset, model)
    model = agent.model.fit(agent.model, agent.dataset)
    sampler = ThompsonSampling((0, 1))
    sampler_ei = ExpectedImprovement((0, 1))
    key, subkey = random.split(key)
    x_next, _ = sampler(model, 1, subkey)
    print(x_next)
    x_next_ei, _ = sampler_ei(model, 1, subkey)
    print(x_next_ei)
    agent.dataset.add_sampled_point(x_next)
    print(datetime.now() - start)
    visualize_prediction_with_samples(
        agent.pred_model, dataset, dataset.X[-1], dataset.unstandardized_Y[-1]
    )
