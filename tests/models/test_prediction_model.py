from jax import random
import optax
import matplotlib.pyplot as plt

from fedbo.agents import BaseAgent
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective
from fedbo.models import MLP
from fedbo.transformations import ObjectiveTransform, MinMaxScaling
from fedbo.training.loss import MSELoss
from fedbo.visualization.model import create_model_prediction_fig

if __name__ == "__main__":
    key = random.key(0)
    num_samples = 64
    in_dim = 1
    bounds = (-5, 10)
    key, k1, k2 = random.split(key, 3)
    dataset = ObjectiveDataset(
        AckleyObjective(in_dim),
        num_samples,
        k1,
        t=ObjectiveTransform(x_transform_post=MinMaxScaling(bounds, k2)),
    )

    num_epochs = 100
    batch_size = 64
    optimizer = optax.adamw(0.001)
    loss_func = MSELoss()
    key, subkey = random.split(key)
    model = MLP.init_from_config(1, 32, 1, 3, subkey)

    agent = BaseAgent(dataset, model)
    agent.model = MLP.fit(
        agent.model, agent.dataset, num_epochs, batch_size, optimizer, loss_func, subkey
    )
    print(dataset.X[-1], agent.dataset.Y[-1], agent.dataset.unstandardized_Y[-1])
    fig = create_model_prediction_fig(
        agent.pred_model, dataset, dataset.X[-1], agent.dataset.unstandardized_Y[-1]
    )
    plt.show()
