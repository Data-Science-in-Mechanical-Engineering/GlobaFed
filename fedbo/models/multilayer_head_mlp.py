from __future__ import annotations
import os
from pathlib import Path
import pickle
import jax
from jax import random
from jaxtyping import Array, Key
import equinox as eqx
from functools import partial
from typing import TYPE_CHECKING, Type

from fedbo.models.base_model import BaseModel
from fedbo.training import train_step
from fedbo.data.utils import DataLoader


if TYPE_CHECKING:
    import optax
    from fedbo.training.loss import BaseLoss
    from fedbo.data import ObjectiveDataset


class MultiLayerHeadMLP(eqx.Module, BaseModel):
    # Learnable
    backbone: list[eqx.nn.Linear]
    head: list[eqx.nn.Linear]

    # Static
    # Backbone
    in_features: int
    backbone_hidden_features: int
    out_features: int
    num_backbone_layers: int
    # Head
    head_hidden_features: int
    num_head_layers: int

    def __init__(self, backbone: list[eqx.nn.Linear], head: list[eqx.nn.Linear]):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.in_features = self.backbone[0].in_features
        self.backbone_hidden_features = self.backbone[0].out_features
        self.out_features = self.head[-1].out_features
        self.num_backbone_layers = len(self.backbone) - 1
        self.num_head_layers = len(self.head)
        self.head_hidden_features = self.head[0].out_features

    def __call__(self, x: Array) -> Array:
        for layer in self.backbone:
            x = jax.nn.relu(layer(x))
        for layer in self.head[:-1]:
            x = jax.nn.relu(layer(x))
        return self.head[-1](x)

    @classmethod
    def init_from_config(
        cls: Type[MultiLayerHeadMLP],
        in_features: int,
        backbone_hidden_features: int,
        head_hidden_features: int,
        out_features: int,
        num_backbone_layers: int,
        num_head_layers: int,
        key: Key,
    ) -> MultiLayerHeadMLP:
        nun_total_layers = 1 + num_backbone_layers + num_head_layers
        keys = random.split(key, nun_total_layers)
        backbone_keys = keys[: 1 + num_backbone_layers]
        head_keys = keys[1 + num_backbone_layers :]

        # Input Layer
        backbone = [eqx.nn.Linear(in_features, backbone_hidden_features, key=backbone_keys[0])]
        # Backbone Hidden Layers
        backbone.extend(eqx.nn.Linear(backbone_hidden_features, backbone_hidden_features, key=k) for k in backbone_keys[1:])
        # Head Layers
        head = []
        if num_head_layers == 1:
            # Output Layer
            head.append(eqx.nn.Linear(backbone_hidden_features, out_features, key=head_keys[0]))
        else:
            # First head layer
            head.append(eqx.nn.Linear(backbone_hidden_features, head_hidden_features, key=head_keys[0]))
            # Intermediate head layers
            for i in range(1, num_head_layers - 1):
                head.append(eqx.nn.Linear(head_hidden_features, head_hidden_features, key=head_keys[i]))
            # Output Layer
            head.append(eqx.nn.Linear(head_hidden_features, out_features, key=head_keys[-1]))
        return cls(backbone, head)

    def reset(self, key: Key) -> MultiLayerHeadMLP:
        return MultiLayerHeadMLP.init_from_config(**self.extract_config(), key=key)

    def extract_config(self) -> dict[str, int]:
        return {
            "in_features": self.in_features,
            "backbone_hidden_features": self.backbone_hidden_features,
            "head_hidden_features": self.head_hidden_features,
            "out_features": self.out_features,
            "num_backbone_layers": self.num_backbone_layers,
            "num_head_layers": self.num_head_layers,
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
        model: MultiLayerHeadMLP,
        dataset: ObjectiveDataset,
        num_epochs: int,
        batch_size: int,
        optimizer: optax.GradientTransformation,
        loss_function: BaseLoss,
        key: Key,
        head_only: bool = False,
        verbose: bool = False,
        val_freq: int = 100,
    ) -> MultiLayerHeadMLP:
        if head_only:
            false_model = jax.tree_util.tree_map(lambda _: False, model)
            head_spec = jax.tree_util.tree_map(eqx.is_array, model.head)
            filter_spec = eqx.tree_at(lambda tree: tree.head, false_model, replace=head_spec)
        else:
            filter_spec = eqx.is_array

        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        for epoch in range(1, num_epochs + 1):
            running_loss = []
            train_data = DataLoader(dataset, batch_size, key, shuffle=True)
            for x, y in train_data:
                model, opt_state, loss = train_step(model, optimizer, opt_state, loss_function, x, y, filter_spec)
                running_loss.append(loss)
            if verbose and (epoch % val_freq == 0 or epoch == 1):
                print(f"Epoch: {epoch},  loss: {jax.numpy.mean(jax.numpy.array(running_loss))}")
        return model
