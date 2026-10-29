import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from jaxtyping import Array, Key

from fedbo.models.vbll.distributions import Normal, DenseNormal, diag_embed


class Regression(eqx.Module):
    # Learnable parameters
    W_mean: Array
    W_logdiag: Array
    W_offdiag: Array
    noise_logdiag: Array
    noise_mean: Array

    # Static attributes
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    parameterization: str = eqx.field(static=True)
    prior_scale: float = eqx.field(static=True)
    wishart_scale: float = eqx.field(static=True)
    dof: float = eqx.field(static=True)

    init_prior_scale: float = eqx.field(static=True)
    init_dof: float = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: Key,
        parameterization: str = "dense",
        prior_scale: float = 1.0,
        wishart_scale: float = 0.1,
        dof: float = 1.0,
    ):
        """
        Variational Bayesian Linear Regression

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        key : Key
            JAX random key
        parameterization : str
            Parameterization of covariance matrix
        prior_scale : float
            Scale of prior covariance matrix
        wishart_scale : float
            Scale of Wishart prior on noise covariance
        dof : float
            Degrees of freedom of Wishart prior on noise covariance
        """
        key_W_mean, key_W_log, key_W_off, key_noise_log = random.split(key, 4)

        # Static attributes
        self.in_features = in_features
        self.out_features = out_features
        self.parameterization = parameterization
        self.prior_scale = prior_scale * (1.0 / in_features)
        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.0) / 2.0

        self.init_prior_scale = prior_scale
        self.init_dof = dof

        # Learnable Parameters
        # Noise distribution
        self.noise_mean = jnp.zeros(out_features)
        self.noise_logdiag = random.normal(key_noise_log, (out_features,)) * jnp.log(wishart_scale)
        # Last layer distribution
        self.W_mean = random.normal(key_W_mean, (out_features, in_features))
        self.W_logdiag = random.normal(key_W_log, (out_features, in_features)) - 0.5 * jnp.log(in_features)

        if parameterization == "diagonal":
            self.W_offdiag = None
        elif parameterization == "dense":
            self.W_offdiag = random.normal(key_W_off, (out_features, in_features, in_features)) / self.in_features
        else:
            raise ValueError(f"Unsupported parameterization: {parameterization!r}")

    @property
    def W(self) -> Normal | DenseNormal:
        """Constructs the weight distribution."""
        cov_diag = jnp.exp(self.W_logdiag)
        if self.parameterization == "diagonal":
            return Normal(self.W_mean, cov_diag)
        elif self.parameterization == "dense":
            tril = jnp.tril(self.W_offdiag, k=-1) + diag_embed(cov_diag)
            return DenseNormal(self.W_mean, tril)
        else:
            raise ValueError(f"Unsupported parameterization: {self.parameterization!r}")

    @property
    def noise(self) -> Normal:
        """Constructs the noise distribution."""
        return Normal(self.noise_mean, jnp.exp(self.noise_logdiag))

    def __call__(self, x: Array) -> Array:
        """The forward pass of the module."""
        return self.predictive(x).loc

    def predictive(self, x: Array) -> Normal:
        """Computes the predictive distribution for inputs x."""
        return (self.W @ x[..., None]).squeeze(-1) + self.noise
