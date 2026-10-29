from abc import ABC, abstractmethod
from functools import partial
import jax
from jax import random
from jaxtyping import Array, Key
from typing import Type

from fedbo.utils.hydra import HydraInit


class BaseTransformation(HydraInit, ABC):
    @abstractmethod
    def __call__(self, X: Array) -> Array:
        pass

    @abstractmethod
    def inverse(self, X: Array) -> Array:
        pass


class Identity(BaseTransformation):
    def __init__(self, key: Key = None):
        pass

    def __call__(self, array: Array) -> Array:
        return identity(array)

    def inverse(self, array: Array) -> Array:
        return identity(array)


@jax.jit
def identity(X: Array) -> Array:
    return X


class ObjectiveTransform(HydraInit):
    def __init__(
        self,
        x_transform_pre: BaseTransformation = Identity(),
        x_transform_post: BaseTransformation = Identity(),
        f_x_transform: BaseTransformation = Identity(),
        y_transform: BaseTransformation = Identity(),
    ):
        self.x_transform_pre = x_transform_pre
        self.x_transform_post = x_transform_post
        self.f_x_transform = f_x_transform
        self.y_transform = y_transform

    @classmethod
    def type_init(
        cls,
        key: Key,
        x_transform_pre: Type[BaseTransformation] = Identity,
        x_transform_post: Type[BaseTransformation] = Identity,
        f_x_transform: Type[BaseTransformation] = Identity,
        y_transform: Type[BaseTransformation] = Identity,
    ):
        keys = random.split(key, 4)
        return cls(
            x_transform_pre(key=keys[0]),
            x_transform_post(key=keys[1]),
            f_x_transform(key=keys[2]),
            y_transform(key=keys[3]),
        )

    @classmethod
    def identity_init(cls, key: Key):
        return cls.type_init(key, Identity, Identity, Identity, Identity)

    @classmethod
    def hydra_init(cls, **kwargs):
        return partial(cls.type_init, **kwargs)
