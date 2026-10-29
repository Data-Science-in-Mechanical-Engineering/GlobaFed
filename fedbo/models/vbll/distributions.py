from __future__ import annotations
import jax
import jax.numpy as jnp
from jaxtyping import Array
import distrax
import warnings
from jax.scipy.linalg import solve_triangular


def tp(M: Array):
    """Transposes the last two dimensions of an array."""
    return jnp.swapaxes(M, -1, -2)


def sym(M: Array):
    """Symmetrizes the last two dimensions of an array."""
    return (M + tp(M)) / 2.0


def diag_embed(x):
    "Jax equivalent for torch.diag_embed"
    x = jnp.atleast_2d(x)
    out = jax.vmap(jnp.diag)(x)
    return out[0] if out.shape[0] == 1 else out


class Normal(distrax.Normal):
    def __init__(self, loc: Array, scale: Array):
        super().__init__(loc=loc, scale=scale)

    @property
    def mean(self):
        return self.loc

    @property
    def var(self):
        return self.scale**2

    @property
    def chol_covariance(self):
        return diag_embed(self.scale)

    @property
    def covariance_diagonal(self):
        return self.var

    @property
    def covariance(self):
        return diag_embed(self.var)

    @property
    def _cov(self):
        var = self._scale**2
        return diag_embed(var)

    @property
    def precision(self):
        return diag_embed(1.0 / self.var)

    @property
    def logdet_covariance(self):
        return 2 * jnp.log(self.scale).sum(-1)

    @property
    def logdet_precision(self):
        return -2 * jnp.log(self.scale).sum(-1)

    @property
    def trace_covariance(self):
        return self.var.sum(-1)

    @property
    def trace_precision(self):
        return (1.0 / self.var).sum(-1)

    def covariance_weighted_inner_prod(self, b: Array, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (self.var[..., None] * (b**2)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b: Array, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((b**2) / self.var[..., None]).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp: Normal | Array):
        if isinstance(inp, Normal):
            new_cov = self.var + inp.var
            return Normal(self.mean + inp.mean, jnp.sqrt(jnp.clip(new_cov, a_min=1e-12)))
        elif isinstance(inp, Array):
            return Normal(self.mean + inp, self.scale)
        else:
            raise NotImplementedError("Distribution addition only implemented for diag covs")

    def __matmul__(self, inp: Array):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp[..., None, :, :], reduce_dim=False)
        return Normal(self.loc @ inp, jnp.sqrt(jnp.clip(new_cov, a_min=1e-12)))

    def squeeze(self, idx: int):
        return Normal(jnp.squeeze(self.loc, idx), jnp.squeeze(self.scale, idx))


class DenseNormal(distrax.MultivariateNormalTri):
    """
    Represents a multivariate normal distribution with a full covariance matrix,
    parameterized by its Cholesky decomposition.
    """

    def __init__(self, loc: Array, scale_tri: Array):
        super().__init__(loc=loc, scale_tri=scale_tri)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        return self.scale_tri

    @property
    def covariance(self):
        return self.scale_tri @ tp(self.scale_tri)

    @property
    def inverse_covariance(self):
        warnings.warn(
            "Direct matrix inverse for dense covariances is O(N^3), consider using eg inverse weighted inner product"
        )
        inv_chol = jnp.linalg.inv(self.scale_tri)
        return tp(inv_chol) @ inv_chol

    @property
    def logdet_covariance(self):
        return 2.0 * jnp.log(jnp.diagonal(self.scale_tri, axis1=-2, axis2=-1)).sum(-1)

    @property
    def trace_covariance(self):
        return (self.scale_tri**2).sum(axis=(-1, -2))

    def covariance_weighted_inner_prod(self, b: Array, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.scale_tri) @ b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b: Array, reduce_dim=True):
        assert b.shape[-1] == 1
        solved = solve_triangular(self.scale_tri, b, lower=True)
        prod = (solved**2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp: Array):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp[..., None, :, :], reduce_dim=False)
        return Normal(self.loc @ inp, jnp.sqrt(jnp.clip(new_cov, a_min=1e-12)))

    def squeeze(self, idx):
        return DenseNormal(jnp.squeeze(self.loc, idx), jnp.squeeze(self.scale_tri, idx))


class MultivariateNormal(distrax.MultivariateNormalDiag):
    """
    Represents a multivariate normal distribution with a diagonal covariance matrix.
    """

    def __init__(self, loc: Array, scale_diag: Array):
        super().__init__(loc=loc, scale_diag=scale_diag)

    @property
    def mean(self):
        return self.loc

    @property
    def var(self):
        return self.scale_diag**2

    @property
    def chol_covariance(self):
        return diag_embed(self.scale_diag)

    @property
    def covariance_diagonal(self):
        return self.var

    @property
    def covariance(self):
        return diag_embed(self.var)

    @property
    def _cov(self):
        var = self._scale_diag**2
        return diag_embed(var)

    @property
    def precision(self):
        return diag_embed(1.0 / self.var)

    @property
    def logdet_covariance(self):
        return 2 * jnp.log(self.scale_diag).sum(-1)

    @property
    def logdet_precision(self):
        return -2 * jnp.log(self.scale_diag).sum(-1)

    @property
    def trace_covariance(self):
        return self.var.sum(-1)

    @property
    def trace_precision(self):
        return (1.0 / self.var).sum(-1)

    def covariance_weighted_inner_prod(self, b: Array, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (self.var[..., None] * (b**2)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b: Array, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((b**2) / self.var[..., None]).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp: Normal | Array):
        if isinstance(inp, Normal):
            new_cov = self.var + inp.var
            return Normal(self.mean + inp.mean, jnp.sqrt(jnp.clip(new_cov, a_min=1e-12)))
        elif isinstance(inp, Array):
            return Normal(self.mean + inp, self.scale_diag)
        else:
            raise NotImplementedError("Distribution addition only implemented for diag covs")

    def __matmul__(self, inp: Array):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp[..., None, :, :], reduce_dim=False)
        return Normal(self.loc @ inp, jnp.sqrt(jnp.clip(new_cov, a_min=1e-12)))

    def squeeze(self, idx: int):
        return Normal(jnp.squeeze(self.loc, idx), jnp.squeeze(self.scale_diag, idx))
