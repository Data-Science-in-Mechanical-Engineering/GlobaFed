from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import TYPE_CHECKING

from fedbo.training.loss.base_loss import BaseLoss
from fedbo.models.vbll import Normal, DenseNormal

if TYPE_CHECKING:
    from jaxtyping import Array
    from fedbo.models import VBLL
    from fedbo.training.centralized_training import MultiHeadedModel


@jax.jit
def gaussian_kl(p: Normal | DenseNormal, q_scale: float):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean**2).sum(-1).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * jnp.log(q_scale) - p.logdet_covariance).sum(-1)
    return 0.5 * (mse_term + trace_term + logdet_term)


class VBLLLoss(BaseLoss):
    def __call__(self, model: VBLL, x: Array, y: Array, regularization_weight: float) -> Array:
        return vbll_loss(model, x, y, jnp.array(regularization_weight))

    @property
    def name(self):
        return "VBLL_Loss"

    # Used for centralized learning
    def centralized_loss(self, model: MultiHeadedModel, head_idx: int, x: Array, y: Array) -> Array:
        regularization_weight = 1 / x.shape[0]
        x = jax.vmap(model.forward_backbone)(x)
        head = model.heads[head_idx]
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


@eqx.filter_jit
def vbll_loss(model: VBLL, x: Array, y: Array, regularization_weight: Array) -> Array:
    x = jax.vmap(model.forward_backbone)(x)
    W = model.head.W
    noise = model.head.noise
    # Predictive log-likelihood term
    pred_density = Normal(loc=(W.mean @ x[..., None]).squeeze(-1), scale=noise.scale)
    pred_likelihood = pred_density.log_prob(y)
    # Trace term from the ELBO
    trace_term = 0.5 * (
        (W.covariance_weighted_inner_prod(jnp.expand_dims(x, -2)[..., None], reduce_dim=False)) * noise.trace_precision
    )
    # KL divergence and Wishart prior terms
    kl_term = gaussian_kl(W, model.head.prior_scale)
    wishart_term = model.head.dof * noise.logdet_precision - 0.5 * model.head.wishart_scale * noise.trace_precision
    # Combine terms to form the ELBO
    total_elbo = jnp.mean(pred_likelihood - trace_term) + regularization_weight * (wishart_term - kl_term)
    return -total_elbo
