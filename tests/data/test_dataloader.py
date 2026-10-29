from jax import random

from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import ackley_objective
from fedbo.data.utils import DataLoader

if __name__ == "__main__":
    key = random.key(0)
    key, subkey = random.split(key)
    dataset = ObjectiveDataset(ackley_objective, 128, (-5, 10), 3, 0, subkey)
    key, subkey = random.split(key)
    dataloader = DataLoader(dataset, batch_size=16, key=subkey, shuffle=True)
    print(dataloader.indices)
    for batch_X, batch_Y in dataloader:
        # Each batch will have shape (16, 1) for X and Y
        print(batch_X.shape, batch_Y.shape)
        break
    print(dataloader.indices)
