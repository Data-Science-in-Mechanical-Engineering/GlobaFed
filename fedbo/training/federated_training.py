from __future__ import annotations
import jax
from jax import random
from jaxtyping import Key
from typing import TYPE_CHECKING

from fedbo.training.base_trainer import BaseTrainer
from fedbo.training.aggregation import FedOpt
from fedbo.training.core import train_agent, assign_backbone

if TYPE_CHECKING:
    from fedbo.agents import BaseAgent


class FederatedTrainer(BaseTrainer):
    def __init__(
        self,
        num_rounds: int,
        train_cfg: dict,
        train_head_cfg: dict,
        aggregator_cfg: dict,
        key: Key,
    ):
        self.num_rounds = num_rounds
        self.train_cfg = train_cfg
        self.train_head_cfg = train_head_cfg
        self.aggregator_cfg = aggregator_cfg
        self.key = key

    def __call__(self, agents: list[BaseAgent]) -> list[BaseAgent]:
        aggregators = [FedOpt(a, **self.aggregator_cfg) for a in agents]
        for _ in range(self.num_rounds):
            self.key, subkey = random.split(self.key)
            train_keys = random.split(subkey, len(agents))
            # Local training
            agents = [train_agent(agent, **self.train_cfg, key=key) for agent, key in zip(agents, train_keys)]
            # Aggregation
            global_backbones = [agg(agents) for agg in aggregators]
            agents = [assign_backbone(agent, backbone) for agent, backbone in zip(agents, global_backbones)]
            # Head training
            self.key, subkey = random.split(self.key)
            train_head_keys = jax.random.split(subkey, len(agents))
            agents = [
                train_agent(agent, **self.train_head_cfg, key=key, head_only=True)
                for agent, key in zip(agents, train_head_keys)
            ]

        return agents


class RayParallelFederatedTrainer(BaseTrainer):
    def __init__(
        self,
        num_rounds: int,
        train_cfg: dict,
        train_head_cfg: dict,
        aggregator_cfg: dict,
        key: Key,
        num_cpus_per_task: float = 1.0,
    ):
        self.num_rounds = num_rounds
        self.train_cfg = train_cfg
        self.train_head_cfg = train_head_cfg
        self.aggregator_cfg = aggregator_cfg
        self.key = key
        self.num_cpus_per_task = num_cpus_per_task

        try:
            import ray

            if not ray.is_initialized():
                ray.init()
            self.ray = ray
        except ImportError:
            raise ImportError("Ray is required for RayParallelFederatedTrainer. Install with: pip install ray")

    def __call__(self, agents: list[BaseAgent]) -> list[BaseAgent]:
        aggregators = [FedOpt(a, **self.aggregator_cfg) for a in agents]

        @self.ray.remote(num_cpus=self.num_cpus_per_task)
        def train_agent_remote(agent, key, train_cfg, head_only=False):
            from fedbo.training.core import train_agent

            return train_agent(agent, **train_cfg, key=key, head_only=head_only)

        for _ in range(self.num_rounds):
            self.key, subkey = random.split(self.key)
            train_keys = random.split(subkey, len(agents))
            # Parallel local training with Ray
            futures = [
                train_agent_remote.remote(agent, key, self.train_cfg, False) for agent, key in zip(agents, train_keys)
            ]
            agents = self.ray.get(futures)
            # Aggregation
            global_backbones = [agg(agents) for agg in aggregators]
            agents = [assign_backbone(agent, backbone) for agent, backbone in zip(agents, global_backbones)]
            # Parallel head training with Ray
            self.key, subkey = random.split(self.key)
            train_head_keys = jax.random.split(subkey, len(agents))
            futures = [
                train_agent_remote.remote(agent, key, self.train_head_cfg, True)
                for agent, key in zip(agents, train_head_keys)
            ]
            agents = self.ray.get(futures)

        return agents
