from __future__ import annotations
import equinox as eqx
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import optax
    from jaxtyping import Array, PyTree
    from fedbo.training.loss import BaseLoss
    from fedbo.models import MLP
    from fedbo.agents import BaseAgent


@eqx.filter_jit
def train_step(
    model: MLP,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_func: BaseLoss,
    x: Array,
    y: Array,
    filter_spec: PyTree,
):
    @eqx.filter_value_and_grad
    def loss_wrapper(diff_model, static_model, x, y):
        model = eqx.combine(diff_model, static_model)
        return loss_func(model, x, y)

    diff_model, static_model = eqx.partition(model, filter_spec)
    loss, grads = loss_wrapper(diff_model, static_model, x, y)
    model, opt_state = optimizer_step(model, opt, opt_state, grads)
    return model, opt_state, loss


@eqx.filter_jit
def optimizer_step(
    model: MLP,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    grads: Array,
) -> tuple[MLP, optax.OptState]:
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state


def train_agent(agent: BaseAgent, **kwargs) -> BaseAgent:
    trained_model = agent.model.fit(model=agent.model, dataset=agent.dataset, **kwargs)
    return agent.init_from_agent(agent, trained_model)


def assign_backbone(agent: BaseAgent, backbone: list[eqx.nn.Linear]) -> BaseAgent:
    aggregated_model = type(agent.model)(backbone, agent.model.head)
    return agent.init_from_agent(agent, aggregated_model)