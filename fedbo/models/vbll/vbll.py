from __future__ import annotations
import os
from pathlib import Path
import pickle
import jax
from jax import random
from jaxtyping import Array, Key
import equinox as eqx
import optax
from functools import partial
from typing import TYPE_CHECKING, Type

from fedbo.models.base_model import BaseModel
from fedbo.models.vbll.layers import Regression
from fedbo.models.vbll.utils import train_step
from fedbo.data.utils import DataLoader


if TYPE_CHECKING:
    from fedbo.training.loss import VBLLLoss
    from fedbo.data import ObjectiveDataset
    from fedbo.models.vbll.distributions import Normal


class VBLL(eqx.Module, BaseModel):
    # Learnable
    backbone: list[eqx.nn.Linear]
    head: Regression

    # Static
    in_features: int
    hidden_features: int
    out_features: int
    num_hidden_layers: int
    reg_factor: float

    def __init__(self, backbone: list[eqx.nn.Linear], head: Regression, reg_factor: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.in_features = self.backbone[0].in_features
        self.hidden_features = self.backbone[0].out_features
        self.out_features = self.head.out_features
        self.num_hidden_layers = len(self.backbone) - 1
        self.reg_factor = reg_factor

    # Returns the mean
    def __call__(self, x: Array) -> Array:
        return self.head(self.forward_backbone(x))

    def forward_backbone(self, x: Array) -> Array:
        for layer in self.backbone:
            x = jax.nn.relu(layer(x))
        return x

    # Returns normal distribution
    def predictive(self, x: Array) -> Normal:
        return self.head.predictive(self.forward_backbone(x))

    @classmethod
    def init_from_config(
        cls: Type[VBLL],
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_hidden_layers: int,
        key: Key,
        reg_factor: float = 1.0,
        parameterization: str = "dense",
        prior_scale: float = 1.0,
        wishart_scale: float = 0.1,
        dof: float = 1.0,
    ) -> VBLL:
        keys = random.split(key, num_hidden_layers + 2)
        # Input Layer
        backbone = [eqx.nn.Linear(in_features, hidden_features, key=keys[0])]
        # Hidden Layers
        backbone.extend(eqx.nn.Linear(hidden_features, hidden_features, key=keys[i + 1]) for i in range(num_hidden_layers))
        # Output Layer
        head = Regression(
            in_features=hidden_features,
            out_features=out_features,
            parameterization=parameterization,
            prior_scale=prior_scale,
            wishart_scale=wishart_scale,
            dof=dof,
            key=keys[-1],
        )
        return cls(backbone, head, reg_factor=reg_factor)

    def reset(self, key: Key, **kwargs) -> VBLL:
        return VBLL.init_from_config(**self.extract_config(), key=key, **kwargs)

    def extract_config(self) -> dict[str, int]:
        return {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "out_features": self.out_features,
            "num_hidden_layers": self.num_hidden_layers,
            "reg_factor": self.reg_factor,
            "parameterization": self.head.parameterization,
            "prior_scale": self.head.init_prior_scale,
            "wishart_scale": self.head.wishart_scale,
            "dof": self.head.init_dof,
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

    @staticmethod
    def fit(
        model: VBLL,
        dataset: ObjectiveDataset,
        num_epochs: int,
        batch_size: int,
        optimizer: optax.GradientTransformation,
        loss_function: VBLLLoss,
        key: Key,
        head_only: bool = False,
        verbose: bool = False,
        val_freq: int = 100,
    ) -> VBLL:
        """Trains the VBLL model on the given dataset."""
        
        filter_spec = (
            eqx.tree_at(
                lambda tree: (
                    tree.head.W_mean,
                    tree.head.W_logdiag,
                    tree.head.W_offdiag,
                    tree.head.noise_logdiag,
                    tree.head.noise_mean,
                ),
                jax.tree_util.tree_map(lambda _: False, model),
                replace=(True, True, True, True, True),
            )
            if head_only
            else eqx.is_array
        )

        reg_weight = model.reg_factor / len(dataset)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        for epoch in range(1, num_epochs + 1):
            running_loss = []
            train_data = DataLoader(dataset, batch_size, key, shuffle=True)
            for x, y in train_data:
                model, opt_state, loss = train_step(
                    model, optimizer, opt_state, loss_function, x, y, filter_spec, reg_weight
                )
                running_loss.append(loss)
            if verbose and (epoch % val_freq == 0 or epoch == 1):
                print(f"Epoch: {epoch},  loss: {jax.numpy.mean(jax.numpy.array(running_loss))}")
        return model
