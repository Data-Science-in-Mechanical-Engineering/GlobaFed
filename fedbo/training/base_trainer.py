from __future__ import annotations
from abc import ABC, abstractmethod
from jaxtyping import Key
from typing import TYPE_CHECKING

from fedbo.utils.hydra import HydraInit

if TYPE_CHECKING:
    from fedbo.agents import BaseAgent


class BaseTrainer(HydraInit, ABC):
    @abstractmethod
    def __call__(self, agents: list[BaseAgent]) -> list[BaseAgent]:
        pass
