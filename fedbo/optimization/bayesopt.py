from __future__ import annotations
import jax
from jax import random
from typing import TYPE_CHECKING

from fedbo.utils.hydra import HydraInit
from fedbo.metrics import (
    BaseMetric,
    InstantaneousRegret,
    SimpleRegret,
    CumulativeRegret,
)

if TYPE_CHECKING:
    from jaxtyping import Key
    from fedbo.agents import BaseAgent
    from fedbo.training import BaseTrainer, FederatedTrainer
    from fedbo.logger import BaseLogger
    from fedbo.optimization.acquisition import BaseAcquisitionFunc


class BayesOpt(HydraInit):
    def __init__(
        self,
        num_iterations: int,
        agents: list[BaseAgent],
        trainer: BaseTrainer,
        acquisition: BaseAcquisitionFunc,
        logger: BaseLogger,
        key: Key,
        reset_model_freq: int = 1,
        log_acquisition: bool = False,
        clear_caches_freq: int = 50,
    ):
        self.key = key
        self.num_iterations = num_iterations
        self.agents = agents
        self.trainer = trainer
        self.acquisition_function = acquisition
        self.logger = logger
        self.reset_model_freq = reset_model_freq
        self.log_acquisition = log_acquisition
        self.clear_caches_freq = clear_caches_freq
        self.num_agents = len(self.agents)
        self.metrics: list[BaseMetric] = [
            InstantaneousRegret(),
            SimpleRegret(),
            CumulativeRegret(),
        ]
        # Used for logging
        self.iteration_regrets = {metric.name: {i: [] for i in range(num_iterations)} for metric in self.metrics}

    def run(self):
        for i in range(self.num_iterations):
            # Training
            self.agents = self.trainer(self.agents)
            self.logger.log_info(f"Finished training of iteration {i+1}/{self.num_iterations}.")

            # Acquisition Function
            self.key, model_key = random.split(self.key)
            for agent in self.agents:
                self.acquisition(agent)
                regret_dict = self.compute_regrets(agent, i)
                self.logger.log_metrics({**regret_dict, "regret_step": i})
                if self.log_acquisition:
                    self.logger.log_acquisition(
                        agent.pred_model,
                        agent.dataset,
                        agent.id,
                        agent.dataset.X[-1],
                        agent.dataset.unstandardized_Y[-1],
                    )
                if (i + 1) % self.reset_model_freq == 0:
                    self.logger.log_info(f"Reset model: Agent {agent.id}")
                    agent.model = agent.model.reset(model_key)

            if (i + 1) % self.clear_caches_freq == 0:
                jax.clear_caches()
                self.logger.log_info(f"Cleared jax caches!")

            # Log iteration end
            self.logger.log_info(f"Finished acquisition of iteration {i+1}/{self.num_iterations}.")

        self.logger.log_overall_regret(self.iteration_regrets)
        self.logger.finish()

    def acquisition(self, agent: BaseAgent):
        self.key, acquisition_key = random.split(self.key)
        x_next, _ = self.acquisition_function(agent.model, agent.dataset.input_dim, acquisition_key)
        agent.dataset.add_sampled_point(x_next)

    def compute_regrets(self, agent: BaseAgent, step: int):
        regret_dict = {}
        for metric in self.metrics:
            metric_value = metric(agent.dataset.f_X, agent.dataset.min_value)
            regret_dict[f"{metric.name}/agent_{agent.id}"] = metric_value
            self.iteration_regrets[metric.name][step].append(metric_value)
        return regret_dict


class BayesOptWarmStart(BayesOpt):
    def __init__(
        self,
        num_iterations: int,
        agents: list[BaseAgent],
        trainer: FederatedTrainer,
        acquisition: BaseAcquisitionFunc,
        logger: BaseLogger,
        warm_start_iter: int,
        key: Key,
        log_dataset: bool = False,
    ):
        super().__init__(num_iterations, agents, trainer, acquisition, logger, key)
        self.warm_start_iter = warm_start_iter
        self.log_dataset = log_dataset

    def run(self):
        for i in range(self.num_iterations):
            # Training
            self.agents = self.trainer(self.agents)
            self.logger.log_info(f"Finished training of iteration {i+1}/{self.num_iterations}.")

            # Acquisition Function
            self.key, model_key = random.split(self.key)
            for agent in self.agents:
                self.acquisition(agent)
                regret_dict = self.compute_regrets(agent, i)
                self.logger.log_metrics({**regret_dict, "regret_step": i})
                self.logger.log_acquisition(
                    agent.pred_model,
                    agent.dataset,
                    agent.id,
                    agent.dataset.X[-1],
                    agent.dataset.unstandardized_Y[-1],
                )
                agent.model = agent.model.reset(model_key)

            if i + 1 == self.warm_start_iter:
                ref_agent = next(iter(self.agents))
                agent_type = type(ref_agent)
                dataset_type = type(ref_agent.dataset)
                dataset_cfg = ref_agent.dataset.config
                self.key, dataset_key = random.split(self.key)
                dataset = dataset_type(**dataset_cfg, key=dataset_key)
                model = ref_agent.model.reset(model_key)
                self.agents.append(agent_type(dataset, model))

            # Log iteration end
            self.logger.log_info(f"Finished acquisition of iteration {i+1}/{self.num_iterations}.")

        self.logger.log_overall_regret(self.iteration_regrets)
        if self.log_dataset:
            for a in self.agents:
                self.logger.log_dataset(a.dataset, a.id)
        self.logger.finish()
