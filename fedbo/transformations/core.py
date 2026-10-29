import jax
from jax import random
from jaxtyping import Array, Key

from fedbo.transformations.base import BaseTransformation


# x_pre or fx
class RandomShifting(BaseTransformation):
    """Randomly shifts the array by a constant factor."""

    def __init__(self, shift_bounds: tuple[int, int], key: Key):
        lb, ub = shift_bounds
        self.shift_factor = random.uniform(key, minval=lb, maxval=ub)

    def __call__(self, X: Array) -> Array:
        return add(X, self.shift_factor)

    def inverse(self, X: Array) -> Array:
        return subtract(X, self.shift_factor)


# x_post
class MinMaxScaling(BaseTransformation):
    """Applies Min-Max scaling to transform array to [0, 1] range."""

    def __init__(self, dataset_bounds: tuple[int, int], key: Key = None):
        self.lb, self.ub = dataset_bounds

    def __call__(self, X: Array) -> Array:
        return minmax_scaling(X, self.lb, self.ub)

    def inverse(self, X: Array) -> Array:
        return inverse_minmax_scaling(X, self.lb, self.ub)


# fx
class RandomScaling(BaseTransformation):
    """Randomly scales the array by a constant factor."""

    def __init__(self, scale_bounds: tuple[int, int], key: Key):
        lb, ub = scale_bounds
        self.scale_factor = random.uniform(key, minval=lb, maxval=ub)

    def __call__(self, X: Array) -> Array:
        return mul(X, self.scale_factor)

    def inverse(self, X: Array) -> Array:
        return mul(X, 1 / self.scale_factor)


# y
class GaussianNoise(BaseTransformation):
    """Adds gaussian noise to the array."""

    def __init__(self, scale: float, key: Key):
        self.scale = scale
        self.noise = None
        self.key = key

    def __call__(self, X: Array) -> Array:
        self.key, subkey = random.split(self.key)
        self.noise = self.scale * random.normal(subkey, shape=X.shape)
        return add(X, self.noise)

    def inverse(self, X: Array) -> Array:
        return subtract(X, self.noise)


@jax.jit
def add(X1: Array, X2: Array) -> Array:
    return X1 + X2


@jax.jit
def subtract(X1: Array, X2: Array) -> Array:
    return X1 - X2


@jax.jit
def mul(X1: Array, X2: Array) -> Array:
    return X1 * X2


@jax.jit
def standardize(X: Array, mean: float, std: float) -> Array:
    """Standardizes X to have zero mean and unit variance."""
    return (X - mean) / std


@jax.jit
def unstandardize(Z: Array, mean: float, std: float) -> Array:
    """Reverts the standardization."""
    return Z * std + mean


@jax.jit
def minmax_scaling(X: Array, lb: float, ub: float) -> Array:
    """Applies Min-Max scaling to transform X to [0, 1] range."""
    return (X - lb) / (ub - lb)


@jax.jit
def inverse_minmax_scaling(X_scaled: Array, lb: float, ub: float) -> Array:
    """Maps values that were scaled to [0, 1] back to the original [lb, ub] range."""
    return X_scaled * (ub - lb) + lb
