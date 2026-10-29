from __future__ import annotations
from abc import ABC, abstractmethod
from jaxtyping import Array, Key
from typing import TYPE_CHECKING

from fedbo.utils.hydra import HydraInit

if TYPE_CHECKING:
    from fedbo.data import ObjectiveDataset


class BaseModel(HydraInit, ABC):
    @abstractmethod
    def __call__(self, x: Array) -> Array:
        pass

    @abstractmethod
    def reset(self, key: Key) -> BaseModel:
        pass

    @staticmethod
    @abstractmethod
    def fit(model: BaseModel, dataset: ObjectiveDataset, key: Key, **kwargs) -> BaseModel:
        pass
