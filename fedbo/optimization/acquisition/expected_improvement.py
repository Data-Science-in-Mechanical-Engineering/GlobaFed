import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim.optimize import optimize_acqf
import jax.numpy as jnp
from jax import random
from jaxtyping import Key

from fedbo.optimization.acquisition.base_acquisition import BaseAcquisitionFunc
from fedbo.models import GaussianProcess


class ExpectedImprovement(BaseAcquisitionFunc):
    def __init__(self, bounds: tuple[int, int] = (0, 1)):
        self.bounds = bounds

    def __call__(self, model: GaussianProcess, input_dim: int, key: Key):
        seed = int(random.randint(key, shape=(), minval=0, maxval=2**16 - 1))
        torch.manual_seed(seed)

        gp_model = model.gp
        min_f = gp_model.train_targets.min()
        log_ei = LogExpectedImprovement(model.gp, best_f=min_f, maximize=False)
        bounds_tensor = torch.tensor(
            [
                [self.bounds[0]] * input_dim,
                [self.bounds[1]] * input_dim,
            ],
            dtype=torch.float32,
        )
        candidate, acq_value = optimize_acqf(
            acq_function=log_ei,
            bounds=bounds_tensor,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        next_point = jnp.array(candidate.detach().numpy()).reshape(-1)

        return next_point, acq_value
