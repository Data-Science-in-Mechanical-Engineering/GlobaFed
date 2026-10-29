from __future__ import annotations
from abc import ABC, abstractmethod
from jaxtyping import Array, Key
from typing import Callable




class BaseAcquisitionFunc(ABC):
    @abstractmethod
    def __call__(self, model: Callable[[Array], Array], key: Key):
        pass
