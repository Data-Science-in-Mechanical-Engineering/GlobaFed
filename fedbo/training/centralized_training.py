from __future__ import annotations
import jax
from jax import random
import equinox as eqx
import optax
from jaxtyping import Key, Array
from typing import TYPE_CHECKING

from fedbo.training.base_trainer import BaseTrainer
from fedbo.training.core import train_agent
from fedbo.data.utils import DataLoader

if TYPE_CHECKING:
    from fedbo.agents import BaseAgent
    from fedbo.data import ObjectiveDataset
    from fedbo.models import MLP, VBLL
    from fedbo.training.loss import BaseLoss


class CentralizedTrainer(BaseTrainer):
    def __init__(
        self,
        num_rounds: int,
        local_train_cfg: dict,
        global_train_cfg: dict,
        head_train_cfg: dict,
        key: Key,
    ):
        self.num_rounds = num_rounds
        self.local_train_cfg = local_train_cfg
        self.global_train_cfg = global_train_cfg
        self.head_train_cfg = head_train_cfg
        self.key = key

    def __call__(self, agents: list[BaseAgent]) -> list[BaseAgent]:
        for _ in range(self.num_rounds):
            # Local training
            self.key, subkey = random.split(self.key)
            train_keys = random.split(subkey, len(agents))
            agents = [train_agent(agent, **self.local_train_cfg, key=key) for agent, key in zip(agents, train_keys)]
            # Central backbone training
            self.key, subkey = random.split(self.key)
            global_model = train_global_model(
                *zip(*((a.model, a.dataset) for a in agents)),
                **self.global_train_cfg,
                key=subkey,
            )
            agents = [
                agent.init_from_agent(
                    agent,
                    type(agent.model)(global_model.backbone, global_model.heads[i]),
                )
                for i, agent in enumerate(agents)
            ]
            # Head training
            self.key, subkey = random.split(self.key)
            train_head_keys = jax.random.split(subkey, len(agents))
            agents = [
                train_agent(agent, **self.head_train_cfg, key=key, head_only=True)
                for agent, key in zip(agents, train_head_keys)
            ]

        return agents


class MultiHeadedModel(eqx.Module):
    backbone: list[eqx.nn.Linear]
    heads: list[eqx.Module]

    def __call__(self, x: Array, head_idx: int) -> Array:
        # Shared backbone computation
        for layer in self.backbone:
            x = jax.nn.relu(layer(x))

        # Task-specific head
        return self.heads[head_idx](x)

    def predict_vmap(self, X: Array, head_idx: int) -> Array:
        return jax.vmap(self, in_axes=(0, None))(X, head_idx)

    def forward_backbone(self, x: Array) -> Array:
        for layer in self.backbone:
            x = jax.nn.relu(layer(x))
        return x

    def forward_backbone_vmap(self, X: Array) -> Array:
        return jax.vmap(self.forward_backbone)(X)


def train_global_model(
    models: list[MLP] | list[VBLL],
    datasets: list[ObjectiveDataset],
    num_epochs: int,
    batch_size: int,
    optimizer: optax.GradientTransformation,
    loss_function: BaseLoss,
    key: Key,
) -> MultiHeadedModel:
    # Init global model
    backbones = [m.backbone for m in models]
    avg_backbone = jax.tree.map(lambda *xs: sum(xs) / len(xs), *backbones)
    heads = [m.head for m in models]
    global_model = MultiHeadedModel(avg_backbone, heads)

    d_keys = random.split(key, len(models))
    dataloaders = [DataLoader(d, batch_size, k, True) for d, k in zip(datasets, d_keys)]
    opt_state = optimizer.init(eqx.filter(global_model, eqx.is_array))
    for epoch in range(num_epochs):
        for batches_tuple in zip(*dataloaders):
            xs = [batch[0] for batch in batches_tuple]
            ys = [batch[1] for batch in batches_tuple]
            global_model, opt_state, loss = train_step(global_model, optimizer, opt_state, xs, ys, loss_function)
    return global_model


@eqx.filter_jit
def train_step(
    model: MultiHeadedModel,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    xs: list[Array],  # List of input batches
    ys: list[Array],  # List of target batches
    base_loss_fn: BaseLoss,
) -> tuple[MultiHeadedModel, optax.OptState, Array]:

    loss_for_grad = lambda m: global_loss(m, xs, ys, base_loss_fn)
    loss, grads = eqx.filter_value_and_grad(loss_for_grad)(model)

    # `grads` will have gradients for the shared backbone and all heads.
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


def global_loss(
    model: MultiHeadedModel,
    xs: list[Array],  # A list of x-batches
    ys: list[Array],  # A list of y-batches
    loss_func: BaseLoss,
) -> Array:
    total_loss = 0.0
    num_heads = len(model.heads)
    # Calculate loss for each task and accumulate
    for i in range(num_heads):
        # Calculate the loss for the current head
        head_loss = loss_func.centralized_loss(model, i, xs[i], ys[i])
        total_loss = total_loss + head_loss

    # Return the average loss
    return total_loss / num_heads
