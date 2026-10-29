import jax
import jax.numpy as jnp
from jax import random
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from botorch.generation import MaxPosteriorSampling
from jaxtyping import Array, Key
from typing import Callable

from fedbo.optimization.acquisition.base_acquisition import BaseAcquisitionFunc
from fedbo.models import (
    MLP,
    MultiLayerHeadMLP,
    VBLL,
    MultiLayerHeadVBLL,
    VBLLSampleModel,
    GaussianProcess,
    NegateOutputModel,
)
from fedbo.models.vbll import MLHVBLLSampleModel
from fedbo.utils.minimize import minimize_func


class ThompsonSampling(BaseAcquisitionFunc):
    def __init__(self, bounds: tuple[int, int] = (0, 1)):
        self.bounds = bounds
        self.fn = {
            MLP: minimize_func,
            MultiLayerHeadMLP: minimize_func,
            GaussianProcess: minimize_gp,
            VBLL: minimize_vbll,
            MultiLayerHeadVBLL: minimize_mlh_vbll,
        }

    def __call__(self, model: Callable[[Array], Array], input_dim: int, key: Key):
        return self.fn[type(model)](model, input_dim, self.bounds, key)


def minimize_gp(
    model: GaussianProcess,
    input_dim: int,
    bounds: tuple[int, int],
    key: Key,
    n_candidates: int = 5000,
    num_samples: int = 1,
):
    lb, ub = bounds

    key, subkey = random.split(key)
    seed = int(random.randint(subkey, shape=(), minval=0, maxval=2**16 - 1))
    sobol = SobolEngine(input_dim, scramble=True, seed=seed)
    X_Cand = sobol.draw(n_candidates)
    X_Cand = lb + X_Cand * (ub - lb)

    thompson_sampling = MaxPosteriorSampling(
        model=NegateOutputModel(model.gp),
        replacement=False,
    )
    with torch.no_grad():  # We don't need gradients when using TS
        X_next: Tensor = thompson_sampling(X_Cand, num_samples=num_samples)
    return jnp.squeeze(X_next.detach().numpy(), axis=0), None


def minimize_vbll(
    model: VBLL,
    input_dim: int,
    bounds: tuple[int, int],
    key: Key,
    num_samples: int = 1,
):
    keys = random.split(key, num_samples)

    def minimize_sample(k: Key) -> tuple[Array, Array]:
        sample_key, optim_key = random.split(k)
        sample_model = VBLLSampleModel(model, sample_key)
        return minimize_func(sample_model, input_dim, bounds, optim_key)

    batch_best_x, batch_best_y = jax.vmap(minimize_sample)(keys)
    best_idx = jnp.argmin(batch_best_y)
    best_x = batch_best_x[best_idx]
    best_y = batch_best_y[best_idx]
    return best_x, best_y


def minimize_mlh_vbll(
    model: MultiLayerHeadVBLL,
    input_dim: int,
    bounds: tuple[int, int],
    key: Key,
    num_samples: int = 1,
):
    keys = random.split(key, num_samples)

    def minimize_sample(k: Key) -> tuple[Array, Array]:
        sample_key, optim_key = random.split(k)
        sample_model = MLHVBLLSampleModel(model, sample_key)
        return minimize_func(sample_model, input_dim, bounds, optim_key)

    batch_best_x, batch_best_y = jax.vmap(minimize_sample)(keys)
    best_idx = jnp.argmin(batch_best_y)
    best_x = batch_best_x[best_idx]
    best_y = batch_best_y[best_idx]
    return best_x, best_y
