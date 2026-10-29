from __future__ import annotations
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.fit import fit_gpytorch_mll
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from jax import random
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key
from typing import TYPE_CHECKING

from fedbo.models.base_model import BaseModel

if TYPE_CHECKING:
    from fedbo.data import ObjectiveDataset


class GaussianProcess(BaseModel):
    def __init__(self, key: Key):
        self.gp: SingleTaskGP = None
        key, subkey = random.split(key)
        seed = int(random.randint(subkey, shape=(), minval=0, maxval=2**16 - 1))
        torch.manual_seed(seed)

    def __call__(self, x: Array) -> Array:
        x = torch.from_numpy(np.asarray(x).copy())
        return jnp.array(self.gp.posterior(x).mean.detach().numpy())

    def sample(self, x: Array) -> Array:
        x = torch.from_numpy(np.asarray(x).copy())
        val = self.gp.posterior(x).rsample(sample_shape=torch.Size([1]))
        return jnp.array(val.detach().numpy())

    @staticmethod
    def fit(
        model: GaussianProcess, dataset: ObjectiveDataset, key: Key = None
    ) -> GaussianProcess:
        train_X = torch.from_numpy(np.array(dataset.X))
        train_Y = torch.from_numpy(np.array(dataset.Y))

        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=dataset.input_dim,
                lengthscale_constraint=Interval(0.005, 4.0),
            )
        )
        gp = SingleTaskGP(
            train_X, train_Y, covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        model.gp = gp
        return model

    def reset(self, key: Key):
        return type(self)(key)


class NegateOutputModel(Model):
    def __init__(self, base_model: Model):
        super().__init__()
        self.base_model = base_model

    def posterior(self, X: Tensor, **kwargs):
        base_posterior = self.base_model.posterior(X, **kwargs)
        cov = base_posterior.mvn.covariance_matrix
        jitter = 1e-6 * torch.eye(cov.size(-1), device=cov.device)
        stable_cov = cov + jitter
        inverted_mvn = MultivariateNormal(
            loc=-base_posterior.mvn.mean,
            covariance_matrix=stable_cov,
        )
        return type(base_posterior)(inverted_mvn)

    def condition_on_observations(self, X, Y, **kwargs):
        conditioned = self.base_model.condition_on_observations(X, -Y, **kwargs)
        return NegateOutputModel(conditioned)

    def subset_output(self, indices: list[int]):
        return NegateOutputModel(self.base_model.subset_output(indices))

    @property
    def num_outputs(self):
        return self.base_model.num_outputs
