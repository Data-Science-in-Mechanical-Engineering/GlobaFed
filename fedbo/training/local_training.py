from __future__ import annotations
import jax
from jax import random
from jaxtyping import Key
from typing import TYPE_CHECKING

from fedbo.training.base_trainer import BaseTrainer
from fedbo.training.core import train_agent

if TYPE_CHECKING:
    from fedbo.agents import BaseAgent


class LocalTrainer(BaseTrainer):
    def __init__(
        self,
        train_cfg: dict,
        key: Key,
    ):
        self.train_cfg = train_cfg
        self.key = key

    def __call__(self, agents: list[BaseAgent]) -> list[BaseAgent]:
        self.key, subkey = random.split(self.key)
        train_keys = jax.random.split(subkey, len(agents))
        # Local training
        agents = [
            train_agent(agent, **self.train_cfg, key=key)
            for agent, key in zip(agents, train_keys)
        ]
        return agents
