from __future__ import annotations
from jax import random
import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyping import Array, Key
    from fedbo.data import ObjectiveDataset


class DataLoader:
    def __init__(self, dataset: ObjectiveDataset, batch_size: int, key: Key, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.key = key
        self.shuffle = shuffle
        self.indices = jnp.arange(len(dataset))

    def update_dataset(self, dataset: ObjectiveDataset):
        self.dataset = dataset
        self.indices = jnp.arange(len(dataset))
        if self.shuffle:
            self.indices = self.shuffle_array(self.indices)

    def shuffle_array(self, array: Array) -> Array:
        self.key, subkey = random.split(self.key)
        return random.permutation(subkey, array)

    def __iter__(self):
        # Shuffle indices if required
        if self.shuffle:
            self.indices = self.shuffle_array(self.indices)

        # Iterate over the data in batches
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch_X = jnp.take(self.dataset.X, batch_indices, axis=0)
            batch_Y = jnp.take(self.dataset.Y, batch_indices, axis=0)

            yield batch_X, batch_Y
