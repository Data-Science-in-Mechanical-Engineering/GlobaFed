import torch
import jax
import jax.numpy as jnp
import distrax
import numpy as np
from jax.scipy.linalg import solve_triangular
from numpy.testing import assert_allclose
from jaxtyping import Array
import warnings

from fedbo.models.vbll import Normal, DenseNormal


def compare_tensors(pt_tensor: torch.Tensor, jax_array: Array, name, atol=1e-6, rtol=1e-6):
    """Compares a PyTorch tensor and a JAX array for equality."""
    pt_numpy = pt_tensor.cpu().detach().numpy()
    jax_numpy = np.array(jax_array)

    try:
        assert_allclose(pt_numpy, jax_numpy, atol=atol, rtol=rtol)
        print(f"PASSED: {name}")
    except AssertionError as e:
        print(f"FAILED: {name}")
        print(e)


# =============================================================================
# Original PyTorch Implementation
# =============================================================================
def pt_tp(M: torch.Tensor):
    return M.transpose(-1, -2)


class TorchNormal(torch.distributions.Normal):
    def __init__(self, loc, chol):
        super(TorchNormal, self).__init__(loc, chol)

    @property
    def mean(self):
        return self.loc

    @property
    def var(self):
        return self.scale**2

    @property
    def chol_covariance(self):
        return torch.diag_embed(self.scale)

    @property
    def covariance(self):
        return torch.diag_embed(self.var)

    @property
    def logdet_covariance(self):
        return 2 * torch.log(self.scale).sum(-1)

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (self.var.unsqueeze(-1) * (b**2)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp):
        if isinstance(inp, TorchNormal):
            new_cov = self.var + inp.var
            return TorchNormal(self.mean + inp.mean, torch.sqrt(torch.clip(new_cov, min=1e-12)))
        elif isinstance(inp, torch.Tensor):
            return TorchNormal(self.mean + inp, self.scale)
        else:
            raise NotImplementedError()

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim=False)
        return TorchNormal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))


class TorchDenseNormal(torch.distributions.MultivariateNormal):
    def __init__(self, loc, cholesky):
        super(TorchDenseNormal, self).__init__(loc, scale_tril=cholesky)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        return self.scale_tril

    @property
    def covariance(self):
        return self.scale_tril @ pt_tp(self.scale_tril)

    @property
    def logdet_covariance(self):
        return 2.0 * torch.diagonal(self.scale_tril, dim1=-2, dim2=-1).log().sum(-1)

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((pt_tp(self.scale_tril) @ b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (torch.linalg.solve(self.scale_tril, b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim=False)
        return TorchNormal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))


if __name__ == "__main__":
    print("Starting Tests...")

    # --- Test Setup ---
    key = jax.random.key(0)
    batch_shape = (1,)
    D = 1  # Dimension of the random variable

    # --- Data for Normal ---
    loc_key, scale_key, add_loc_key, add_scale_key, b_key, matmul_key = jax.random.split(key, 6)

    loc_jax = jax.random.normal(loc_key, batch_shape + (D,))
    scale_jax = jax.random.uniform(scale_key, batch_shape + (D,)) + 0.1  # Ensure positive scale
    b_jax = jax.random.normal(b_key, batch_shape + (D, 1))
    matmul_inp_jax = jax.random.normal(matmul_key, batch_shape + (D, 1))

    loc_pt = torch.from_numpy(np.array(loc_jax))
    scale_pt = torch.from_numpy(np.array(scale_jax))
    b_pt = torch.from_numpy(np.array(b_jax))
    matmul_inp_pt = torch.from_numpy(np.array(matmul_inp_jax))

    # --- Data for DenseNormal ---
    loc_dense_jax = loc_jax
    L_jax = jnp.tril(jax.random.normal(key, batch_shape + (D, D)))

    loc_dense_pt = loc_pt
    L_pt = torch.from_numpy(np.array(L_jax))

    # --- Test Normal ---
    print("\n" + "=" * 20 + " Testing Normal Class " + "=" * 20)
    dist_pt = TorchNormal(loc_pt, scale_pt)
    dist_jax = Normal(loc_jax, scale_jax)

    compare_tensors(dist_pt.mean, dist_jax.mean, "Normal.mean")
    compare_tensors(dist_pt.var, dist_jax.var, "Normal.var")
    compare_tensors(dist_pt.chol_covariance, dist_jax.chol_covariance, "Normal.chol_covariance")
    compare_tensors(dist_pt.covariance, dist_jax.covariance, "Normal.covariance")
    compare_tensors(dist_pt.logdet_covariance, dist_jax.logdet_covariance, "Normal.logdet_covariance")
    compare_tensors(
        dist_pt.covariance_weighted_inner_prod(b_pt),
        dist_jax.covariance_weighted_inner_prod(b_jax),
        "Normal.covariance_weighted_inner_prod",
    )

    # Test __add__ with another distribution
    loc2_jax = jax.random.normal(add_loc_key, batch_shape + (D,))
    scale2_jax = jax.random.uniform(add_scale_key, batch_shape + (D,)) + 0.1
    dist_pt_2 = TorchNormal(torch.from_numpy(np.array(loc2_jax)), torch.from_numpy(np.array(scale2_jax)))
    dist_jax_2 = Normal(loc2_jax, scale2_jax)

    add_dist_pt = dist_pt + dist_pt_2
    add_dist_jax = dist_jax + dist_jax_2
    compare_tensors(add_dist_pt.mean, add_dist_jax.mean, "Normal.__add__ (dist) mean")
    compare_tensors(add_dist_pt.scale, add_dist_jax.scale, "Normal.__add__ (dist) scale")

    # Test __add__ with a tensor
    add_tensor_pt = dist_pt + loc_pt
    add_tensor_jax = dist_jax + loc_jax
    compare_tensors(add_tensor_pt.mean, add_tensor_jax.mean, "Normal.__add__ (tensor) mean")

    # Test __matmul__
    matmul_dist_pt = dist_pt @ matmul_inp_pt
    matmul_dist_jax = dist_jax @ matmul_inp_jax
    compare_tensors(matmul_dist_pt.mean, matmul_dist_jax.mean, "Normal.__matmul__ mean")
    compare_tensors(matmul_dist_pt.scale, matmul_dist_jax.scale, "Normal.__matmul__ scale")

    # --- Test DenseNormal ---
    print("\n" + "=" * 20 + " Testing DenseNormal Class " + "=" * 20)
    dist_dense_pt = TorchDenseNormal(loc_dense_pt, L_pt)
    dist_dense_jax = DenseNormal(loc_dense_jax, L_jax)

    compare_tensors(dist_dense_pt.mean, dist_dense_jax.mean, "DenseNormal.mean")
    compare_tensors(dist_dense_pt.chol_covariance, dist_dense_jax.chol_covariance, "DenseNormal.chol_covariance")
    compare_tensors(dist_dense_pt.covariance, dist_dense_jax.covariance, "DenseNormal.covariance")
    compare_tensors(dist_dense_pt.logdet_covariance, dist_dense_jax.logdet_covariance, "DenseNormal.logdet_covariance")
    compare_tensors(
        dist_dense_pt.covariance_weighted_inner_prod(b_pt),
        dist_dense_jax.covariance_weighted_inner_prod(b_jax),
        "DenseNormal.covariance_weighted_inner_prod",
    )
    compare_tensors(
        dist_dense_pt.precision_weighted_inner_prod(b_pt),
        dist_dense_jax.precision_weighted_inner_prod(b_jax),
        "DenseNormal.precision_weighted_inner_prod",
    )

    # Test __matmul__
    matmul_dense_pt = dist_dense_pt @ matmul_inp_pt
    matmul_dense_jax = dist_dense_jax @ matmul_inp_jax
    compare_tensors(matmul_dense_pt.mean, matmul_dense_jax.mean, "DenseNormal.__matmul__ mean")
    compare_tensors(matmul_dense_pt.scale, matmul_dense_jax.scale, "DenseNormal.__matmul__ scale")
    print("All tests completed!")
