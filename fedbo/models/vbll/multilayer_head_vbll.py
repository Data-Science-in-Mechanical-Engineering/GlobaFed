from __future__ import annotations
import os
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Key
import equinox as eqx
import optax
from functools import partial
from typing import TYPE_CHECKING, Type, Any

from fedbo.models.base_model import BaseModel
from fedbo.models.vbll.layers import Regression
from fedbo.models.vbll.distributions import Normal
from fedbo.training.loss.vbll_loss import gaussian_kl

from fedbo.models.vbll.utils import train_step
from fedbo.data.utils import DataLoader


if TYPE_CHECKING:
    from fedbo.training.loss import VBLLLoss
    from fedbo.data import ObjectiveDataset


class MultiLayerHeadVBLL(eqx.Module, BaseModel):
    # Learnable
    backbone: list[eqx.nn.Linear]
    head: list[eqx.Module]

    # Static
    # Backbone
    in_features: int
    backbone_hidden_features: int
    out_features: int
    num_backbone_layers: int
    # Head
    head_hidden_features: int
    num_head_layers: int
    reg_factor: float

    def __init__(self, backbone: list[eqx.nn.Linear], head: list[eqx.Module], reg_factor: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.in_features = self.backbone[0].in_features
        self.backbone_hidden_features = self.backbone[0].out_features
        # Out features from the last layer (must be Regression)
        self.out_features = self.head[-1].out_features
        self.num_backbone_layers = len(self.backbone) - 1
        self.num_head_layers = len(self.head)
        # Head hidden features depends on whether there are intermediate layers
        self.head_hidden_features = self.head[0].out_features if len(self.head) > 1 else self.backbone_hidden_features
        self.reg_factor = reg_factor

    # Returns the mean prediction
    def __call__(self, x: Array) -> Array:
        """Forward pass to compute the mean prediction."""
        x = self.forward_backbone(x)
        return self.head[-1](x)

    def forward_backbone(self, x: Array) -> Array:
        """Forward pass through the backbone layers with ReLU activation."""
        for layer in self.backbone:
            x = jax.nn.relu(layer(x))
        for layer in self.head[:-1]:
            x = jax.nn.relu(layer(x))
        return x

    # Returns normal distribution
    def predictive(self, x: Array) -> Normal:
        """Computes the predictive Normal distribution."""
        x = self.forward_backbone(x)
        return self.head[-1].predictive(x)

    @classmethod
    def init_from_config(
        cls: Type[MultiLayerHeadVBLL],
        in_features: int,
        backbone_hidden_features: int,
        head_hidden_features: int,
        out_features: int,
        num_backbone_layers: int,
        num_head_layers: int,
        key: Key,
        # Regression layer params
        reg_factor: float = 1.0,
        parameterization: str = "dense",
        prior_scale: float = 1.0,
        wishart_scale: float = 0.1,
        dof: float = 1.0,
    ) -> MultiLayerHeadVBLL:
        # Calculate the number of keys needed
        num_total_layers = 1 + num_backbone_layers + num_head_layers
        keys = random.split(key, num_total_layers)
        backbone_keys = keys[: 1 + num_backbone_layers]
        head_keys = keys[1 + num_backbone_layers :]

        # --- Backbone Initialization ---
        # Input Layer
        backbone = [eqx.nn.Linear(in_features, backbone_hidden_features, key=backbone_keys[0])]
        # Backbone Hidden Layers
        backbone.extend(eqx.nn.Linear(backbone_hidden_features, backbone_hidden_features, key=k) for k in backbone_keys[1:])

        # --- Head Initialization ---
        head = []
        head_input_features = backbone_hidden_features

        if num_head_layers == 1:
            # Output Layer: Regression
            head.append(
                Regression(
                    in_features=head_input_features,
                    out_features=out_features,
                    parameterization=parameterization,
                    prior_scale=prior_scale,
                    wishart_scale=wishart_scale,
                    dof=dof,
                    key=head_keys[0],
                )
            )
        else:
            # First head layer (Linear)
            head.append(eqx.nn.Linear(head_input_features, head_hidden_features, key=head_keys[0]))

            # Intermediate head layers (Linear)
            for i in range(1, num_head_layers - 1):
                head.append(eqx.nn.Linear(head_hidden_features, head_hidden_features, key=head_keys[i]))

            # Output Layer (Regression)
            head.append(
                Regression(
                    in_features=head_hidden_features,
                    out_features=out_features,
                    parameterization=parameterization,
                    prior_scale=prior_scale,
                    wishart_scale=wishart_scale,
                    dof=dof,
                    key=head_keys[-1],
                )
            )

        return cls(backbone, head, reg_factor=reg_factor)

    def reset(self, key: Key, **kwargs) -> MultiLayerHeadVBLL:
        """Re-initializes the model with a new key."""
        return MultiLayerHeadVBLL.init_from_config(**self.extract_config(), key=key, **kwargs)

    def extract_config(self) -> dict[str, Any]:
        """Extracts configuration parameters for re-initialization."""
        return {
            "in_features": self.in_features,
            "backbone_hidden_features": self.backbone_hidden_features,
            "head_hidden_features": self.head_hidden_features,
            "out_features": self.out_features,
            "num_backbone_layers": self.num_backbone_layers,
            "num_head_layers": self.num_head_layers,
            "reg_factor": self.reg_factor,
            # Regression layer params from the last head layer
            "parameterization": self.head[-1].parameterization,
            "prior_scale": self.head[-1].init_prior_scale,
            "wishart_scale": self.head[-1].wishart_scale,
            "dof": self.head[-1].init_dof,
        }

    def save_checkpoint(self, path: Path):
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "wb+") as checkpoint_file:
            pickle.dump(self, checkpoint_file)

    @staticmethod
    def load_checkpoint(path: Path):
        with open(path, "rb") as checkpoint_file:
            return pickle.load(checkpoint_file)

    @classmethod
    def hydra_init(cls, **kwargs):
        return partial(cls.init_from_config, **kwargs)

    # Currently does not support loss function parameter
    # Loss class leads to recompilation
    @staticmethod   
    def fit(
        model: MultiLayerHeadVBLL,
        dataset: ObjectiveDataset,
        num_epochs: int,
        batch_size: int,
        optimizer: optax.GradientTransformation,
        loss_function: VBLLLoss,
        key: Key,
        head_only: bool = False,
        verbose: bool = False,
        val_freq: int = 100,
    ) -> MultiLayerHeadVBLL:
        """Trains the model using the VBLL training loop."""

        if head_only:
            filter_spec = jax.tree_util.tree_map(lambda _: False, model)

            def get_head_layer_params(i):
                return lambda tree: (tree.head[i].weight, tree.head[i].bias)

            for i in range(len(model.head) - 1):
                filter_spec = eqx.tree_at(
                    get_head_layer_params(i), filter_spec, replace=(True, True)  # Mark weight and bias as True (trainable)
                )
            filter_spec = eqx.tree_at(
                lambda tree: (
                    tree.head[-1].W_mean,
                    tree.head[-1].W_logdiag,
                    tree.head[-1].W_offdiag,
                    tree.head[-1].noise_logdiag,
                    tree.head[-1].noise_mean,
                ),
                filter_spec,
                replace=(True, True, True, True, True),
            )
        else:
            filter_spec = eqx.is_array

        reg_weight = jnp.array(model.reg_factor / len(dataset))
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        # Uses the train_step suitable for VBLL
        for epoch in range(1, num_epochs + 1):
            running_loss = []
            train_data = DataLoader(dataset, batch_size, key, shuffle=True)
            for x, y in train_data:
                model, opt_state, loss = train_step(model, optimizer, opt_state, vbll_loss, x, y, filter_spec, reg_weight)
                running_loss.append(loss)
            if verbose and (epoch % val_freq == 0 or epoch == 1):
                print(f"Epoch: {epoch}, loss: {jax.numpy.mean(jax.numpy.array(running_loss))}")
        return model


@eqx.filter_jit
def vbll_loss(model: MultiLayerHeadVBLL, x: Array, y: Array, regularization_weight: Array) -> Array:
    x = jax.vmap(model.forward_backbone)(x)
    head = model.head[-1]
    W = head.W
    noise = head.noise
    # Predictive log-likelihood term
    pred_density = Normal(loc=(W.mean @ x[..., None]).squeeze(-1), scale=noise.scale)
    pred_likelihood = pred_density.log_prob(y)
    # Trace term from the ELBO
    trace_term = 0.5 * (
        (W.covariance_weighted_inner_prod(jnp.expand_dims(x, -2)[..., None], reduce_dim=False)) * noise.trace_precision
    )
    # KL divergence and Wishart prior terms
    kl_term = gaussian_kl(W, head.prior_scale)
    wishart_term = head.dof * noise.logdet_precision - 0.5 * head.wishart_scale * noise.trace_precision
    # Combine terms to form the ELBO
    total_elbo = jnp.mean(pred_likelihood - trace_term) + regularization_weight * (wishart_term - kl_term)
    return -total_elbo


class MLHVBLLSampleModel(eqx.Module):
    """
    Model with a fixed backbone and a single sample of weights for the output layer.
    It is used to evaluate a single function drawn from the VBLL model's posterior distribution.
    """

    model: MultiLayerHeadVBLL
    sampled_params: Array

    def __init__(self, model: MultiLayerHeadVBLL, key: Key):
        """
        Initializes the SampleModel.
        Args:
            model: The original VBLL model backbone.
            key: A JAX random key to sample the weights for the final layer.
        """
        self.model = model
        self.sampled_params = model.head[-1].W.sample(seed=key, sample_shape=())

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
