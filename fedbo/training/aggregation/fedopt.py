from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array
from typing import TYPE_CHECKING

from fedbo.training.core import optimizer_step

if TYPE_CHECKING:
    from fedbo.agents import BaseAgent
    from fedbo.models import MLP


class FedOpt:
    def __init__(
        self,
        agent: BaseAgent,
        optimizer: optax.GradientTransformation,
    ):
        self.global_backbone = get_backbone_params(agent.model)
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(self.global_backbone)

    def __call__(self, agents: list[BaseAgent]) -> list[eqx.nn.Linear]:
        backbones = [get_backbone_params(a.model) for a in agents]
        dataset_sizes = jnp.array([a.num_train_data for a in agents])
        self.global_backbone, self.opt_state = aggregate_models(
            self.global_backbone,
            backbones,
            dataset_sizes,
            self.optimizer,
            self.opt_state,
        )
        return self.global_backbone


def get_backbone_params(model: MLP):
    params, _ = eqx.partition(model.backbone, eqx.is_array)
    return params


@eqx.filter_jit
def average_models(models: list[eqx.Module]):
    return jax.tree.map(lambda *xs: sum(xs) / len(xs), *models)


@eqx.filter_jit
def aggregate_models(
    prev_global_model: list[eqx.Module],
    models: list[eqx.Module],
    dataset_sizes: Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    # Compute weighted model deltas
    num_data_total = jnp.sum(dataset_sizes)
    weighted_deltas = [
        jax.tree.map(
            lambda bp, pg: (bp - pg) * (num_data / num_data_total),
            backbone_params,
            prev_global_model,
        )
        for backbone_params, num_data in zip(models, dataset_sizes)
    ]
    # Sum the weighted deltas elementwise
    pseudo_gradient = jax.tree.map(lambda *xs: -sum(xs), *weighted_deltas)
    global_backbone, opt_state = optimizer_step(
        prev_global_model,
        optimizer,
        opt_state,
        pseudo_gradient,
    )
    return global_backbone, opt_state
