from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from fedbo.training.loss.base_loss import BaseLoss
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyping import Array
    from fedbo.models import BaseModel
    from fedbo.training.centralized_training import MultiHeadedModel


class MSELoss(BaseLoss):
    def __call__(self, model: BaseModel, x: Array, y: Array) -> Array:
        return mse_loss(model, x, y)

    @property
    def name(self):
        return "MSE_Loss"

    # Used for centralized learning
    def centralized_loss(self, model: MultiHeadedModel, head_idx: int, x: Array, y: Array) -> Array:
        # The `in_axes` specifies to map over the first arg (x) but not the second (head_index)
        pred_y = jax.vmap(model, in_axes=(0, None))(x, head_idx)
        # Calculate the loss for the current head
        return jnp.mean((pred_y - y) ** 2)


@eqx.filter_jit
def mse_loss(model: BaseModel, x: Array, y: Array) -> Array:
    pred_y = jax.vmap(model)(x)
    return jnp.mean((pred_y - y) ** 2)
