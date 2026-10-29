from __future__ import annotations
import itertools

from fedbo.utils.hydra import HydraInit
from fedbo.data import ObjectiveDataset
from fedbo.models import BaseModel, PredictionModel


class BaseAgent(HydraInit):
    # Global iterator for assigning unique agent ids
    id_iterator = itertools.count()

    def __init__(self, dataset: ObjectiveDataset, model: BaseModel, agent_id: int = None):
        """
        Initializes the BaseAgent.

        Args:
            dataset: The local objective dataset specific to this agent.
            model: The local surrogate model for this agent.
            agent_id: An optional, specific ID to assign to the agent.
                If None, a unique ID will be generated automatically.
        """
        self.id = next(BaseAgent.id_iterator) if agent_id is None else agent_id
        self.dataset = dataset
        self.model = model

    @classmethod
    def init_from_agent(cls, agent: BaseAgent, model: BaseModel):
        """
        Class method to create a new agent instance from an existing one.

        Args:
            agent: The source BaseAgent to copy `dataset` and `id` from.
            model: The new model to assign to the new agent instance.

        Returns:
            A new agent instance.
        """
        return cls(agent.dataset, model, agent.id)

    @property
    def num_train_data(self) -> int:
        """
        Property to get the number of data points in the agent's local dataset.

        Returns:
            The size of the local dataset.
        """
        return len(self.dataset)

    @property
    def pred_model(self):
        """
        Property to get a `PredictionModel` instance to make unstandardized predictions.

        Returns:
            A PredictionModel instance of the local model.
        """
        return PredictionModel(self.model, self.dataset.unstandardize_func)
