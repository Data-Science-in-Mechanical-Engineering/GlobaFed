from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyping import Array
    from fedbo.models import BaseModel
    from fedbo.training.centralized_training import MultiHeadedModel


class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, model: BaseModel, x: Array, y: Array) -> Array:
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def centralized_loss(self, model: MultiHeadedModel, head_idx: int, x: Array, y: Array):
        pass
