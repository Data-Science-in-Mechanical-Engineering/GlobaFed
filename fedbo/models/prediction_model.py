import equinox as eqx
from jaxtyping import Array
from typing import Callable

from fedbo.models.base_model import BaseModel


class PredictionModel:
    def __init__(self, model: BaseModel, pred_transform: Callable[[Array], Array]):
        self._model = model
        self.pred_transform = pred_transform

    def __call__(self, x: Array) -> Array:
        return self.pred_transform(self._model(x))

    @property
    def can_vmap(self):
        return isinstance(self._model, eqx.Module)
