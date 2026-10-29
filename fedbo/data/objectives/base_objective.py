from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Array
from typing import Callable


class BaseObjective(ABC):
    def __init__(self, input_dim: int, min_y: float, max_y: float):
        self.input_dim = input_dim
        self.min_y = min_y
        self.max_y = max_y

    @abstractmethod
    def __call__(self, X: Array):
        pass

    @staticmethod
    @abstractmethod
    def bounds() -> tuple[int, int]:
        pass


class ObjectiveWrapper(BaseObjective):
    def __init__(
        self,
        objective_func: Callable[[Array], Array],
        input_dim: int,
        bounds: tuple[int, int],
        min_y: float,
        max_y: float,
    ):
        self.objective_func = objective_func
        self._bounds = bounds
        super().__init__(input_dim, min_y, max_y)

    def __call__(self, X: Array):
        return self.objective_func(X)

    def bounds(self):
        return self._bounds


class TransformedObjective(BaseObjective):
    def __init__(
        self,
        objective_func: BaseObjective,
        bounds: tuple[int, int],
        rotation_mat: Array = 1,
        shift_mat: Array = 0,
    ):
        self.objective_func = objective_func
        self._bounds = bounds
        self.rotation_mat = rotation_mat
        self.shift_mat = shift_mat
        super().__init__(
            objective_func.input_dim, objective_func.min_y, objective_func.max_y
        )

    def __call__(self, X: Array):
        return self.objective_func(self._rotate(self._shift(X)))

    def _shift(self, X: Array):
        return X - self.shift_mat

    def _rotate(self, x):
        return jnp.dot(self.rotation_mat, x)

    def bounds(self):
        return self._bounds
