from __future__ import annotations
import optax
import equinox as eqx
from jaxtyping import Array, Key, PyTree
from typing import TYPE_CHECKING

from fedbo.training import optimizer_step

if TYPE_CHECKING:
    from fedbo.models.vbll.vbll import VBLL
    from fedbo.training.loss import VBLLLoss


class VBLLSampleModel(eqx.Module):
    """
    Model with a fixed backbone and a single sample of weights for the output layer.
    It is used to evaluate a single function drawn from the VBLL model's posterior distribution.
    """

    model: VBLL
    sampled_params: Array

    def __init__(self, model: VBLL, key: Key):
        """
        Initializes the SampleModel.
        Args:
            model: The original VBLL model backbone.
            key: A JAX random key to sample the weights for the final layer.
        """
        self.model = model
        self.sampled_params = model.head.W.sample(seed=key, sample_shape=())

    @eqx.filter_jit
    def __call__(self, x: Array) -> Array:
        """
        Performs a forward pass through the backbone and the sampled output layer.
        Args:
            x: Input data.
        Returns:
            The model's output for the given input, using the sampled parameters.
        """
        x = self.model.forward_backbone(x)
        return (self.sampled_params @ x[..., None]).squeeze(-1)


@eqx.filter_jit
def train_step(
    model: VBLL,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_func: VBLLLoss,
    x: Array,
    y: Array,
    filter_spec: PyTree,
    reg_weight: Array,
) -> tuple[VBLL, optax.OptState, Array]:
    """Performs a single training step for the VBLL model."""

    @eqx.filter_value_and_grad
    def loss_wrapper(diff_model, static_model, x, y):
        model = eqx.combine(diff_model, static_model)
        return loss_func(model, x, y, reg_weight)

    diff_model, static_model = eqx.partition(model, filter_spec)
    loss, grads = loss_wrapper(diff_model, static_model, x, y)
    model, opt_state = optimizer_step(model, opt, opt_state, grads)
    return model, opt_state, loss


# Hydra Init methdo
def vbll_optimizer(max_norm: float, learning_rate: float, weight_decay: float = 0.0):
    """Initializes the VBLL optimizer with gradient clipping and AdamW."""
    return optax.chain(optax.clip_by_global_norm(max_norm), optax.adamw(learning_rate, weight_decay=weight_decay))
