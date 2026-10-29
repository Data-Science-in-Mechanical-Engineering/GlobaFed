from __future__ import annotations
from dataclasses import dataclass
from jax import random
from typing import Callable, Optional, TYPE_CHECKING

from fedbo.transformations import ObjectiveTransform as ObjTransform
from fedbo.logger import BaseLogger

if TYPE_CHECKING:
    from fedbo.optimization import BayesOpt
    from fedbo.agents import BaseAgent
    from fedbo.data import ObjectiveDataset
    from fedbo.models import BaseModel
    from fedbo.training import BaseTrainer
    from fedbo.optimization.acquisition import BaseAcquisitionFunc


@dataclass
class MainConfig:
    seed: int
    num_agents: int
    bayesopt: Callable[..., BayesOpt]
    agent: Callable[..., BaseAgent]
    dataset: Callable[..., ObjectiveDataset]
    model: Callable[..., BaseModel]
    trainer: Callable[..., BaseTrainer]
    acquisition: BaseAcquisitionFunc
    logger: BaseLogger = BaseLogger()
    transformations: Optional[Callable[..., ObjTransform]] = ObjTransform.identity_init
    hydra: Optional[dict] = None

    def create_optimizer(self):
        bayesopt_key = random.key(self.seed)
        bayesopt_key, model_key, data_key, trainer_key = random.split(bayesopt_key, 4)
        dataset_keys = random.split(data_key, self.num_agents * 2)
        agents = [
            self.agent(
                model=self.model(key=model_key),
                dataset=self.dataset(
                    key=dataset_keys[2 * i],
                    t=self.transformations(key=dataset_keys[2 * i + 1]),
                ),
            )
            for i in range(self.num_agents)
        ]
        trainer = self.trainer(key=trainer_key)

        return self.bayesopt(
            agents=agents,
            trainer=trainer,
            acquisition=self.acquisition,
            logger=self.logger,
            key=bayesopt_key,
        )
