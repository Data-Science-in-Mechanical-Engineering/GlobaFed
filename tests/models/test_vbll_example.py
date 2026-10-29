import jax
import jax.numpy as jnp
from jax import random
import optax
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy

from fedbo.models import MLP
from fedbo.models.vbll import VBLL
from fedbo.data.utils import DataLoader
from fedbo.training.loss import MSELoss, VBLLLoss
from fedbo.visualization.model import *


class SimpleFnDataset(Dataset):
    """The simple function used in DUE/SNGP 1-d regression experiments.
    Based on the implementation presented in:
    https://github.com/y0ast/DUE/blob/main/toy_regression.ipynb
    """

    def __init__(self, num_samples):
        self.num_samples = int(num_samples)
        self.X, self.Y = self.get_data()
        self.bounds = [-1.5, 1.5]

    def get_data(self, noise=0.05, seed=2):
        np.random.seed(seed)

        W = np.random.randn(30, 1)
        b = np.random.rand(30, 1) * 2 * np.pi

        x = 5 * np.sign(np.random.randn(self.num_samples)) + np.random.randn(self.num_samples).clip(-2, 2)
        y = np.cos(W * x + b).sum(0) / 5.0 + noise * np.random.randn(self.num_samples)
        return jnp.array(x[..., None]) / 10, jnp.array(y[..., None])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def viz_data(dataset: SimpleFnDataset):
    plt.scatter(dataset.X, dataset.Y, color="k")
    plt.axis([-1.5, 1.5, -2, 2])
    plt.show()


if __name__ == "__main__":
    key = random.key(5)
    num_samples = 128

    dataset = SimpleFnDataset(num_samples=num_samples)
    # viz_data(dataset)

    # MLP
    # in_features = 1
    # hidden_features = 64
    # out_features = 1
    # num_hidden_layers = 4
    # key, mlp_key = random.split(key)
    # mlp_model = MLP.init_from_config(in_features, hidden_features, out_features, num_hidden_layers, mlp_key)

    # num_epochs = 1000
    # batch_size = 64
    # lr = 3e-3
    # optimizer = optax.adamw(lr, weight_decay=0.0)
    # loss_func = MSELoss()
    # key, train_key = random.split(key)
    # start = datetime.now()
    # mlp_model = mlp_model.fit(mlp_model, dataset, num_epochs, batch_size, optimizer, loss_func, train_key, verbose=True)
    # print(f"Train model: {datetime.now() - start}")
    # visualize_prediction(mlp_model, dataset)

    # VBLL
    in_features = 1
    hidden_features = 64
    out_features = 1
    num_hidden_layers = 4
    reg_factor = 1
    key, vbll_key = random.split(key)
    vbll_model = VBLL.init_from_config(
        in_features, hidden_features, out_features, num_hidden_layers, vbll_key, reg_factor=reg_factor
    )

    num_epochs = 1000
    batch_size = 32
    lr = 1e-3
    optimizer = optax.adamw(lr, weight_decay=0.0)
    loss_func = VBLLLoss()
    key, train_key = random.split(key)
    start = datetime.now()
    vbll_model = vbll_model.fit(vbll_model, dataset, num_epochs, batch_size, optimizer, loss_func, train_key, verbose=True)
    print(f"Train model: {datetime.now() - start}")
    vizualize_vbll(vbll_model, dataset)
