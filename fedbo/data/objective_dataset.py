from __future__ import annotations
import jax.numpy as jnp
import jax
from jax import random
from jaxtyping import Array, Key
import numpy as np
from scipy.stats.qmc import Sobol
from functools import partial


from fedbo.utils.hydra import HydraInit
from fedbo.data.objectives import BaseObjective
from fedbo.transformations import ObjectiveTransform, standardize, unstandardize


class ObjectiveDataset(HydraInit):
    def __init__(
        self,
        objective_func: BaseObjective,
        num_samples: int,
        key: Key,
        t: ObjectiveTransform = None,
    ):
        """
        Intialize Objective Dataset.
        Args:
            objective_func: The objective function. In: (num_samples, input_dim). Out: (num_samples, 1).
            num_samples: The initial number of samples.
            bounds: The bounds of the objective function.
            input_dim: The dimension of the input features.
            min_value: The minimum output value
            key: A PRNG Key.
            t: Optional transformations for the objective function.
        """
        self.key = key
        self.init_num_samples = num_samples
        self.objective_func = objective_func
        self._objective = jax.vmap(objective_func)
        self.bounds = objective_func.bounds()
        self.input_dim = objective_func.input_dim
        self.out_dim = 1
        # Original mean and std to undo standardization
        self.Y_mean = None
        self.Y_std = None
        # Transformations
        self.t = ObjectiveTransform() if t is None else t
        self.x_transform_pre = self.t.x_transform_pre
        self.x_transform_post = self.t.x_transform_post
        self.f_x_transform = self.t.f_x_transform
        self.y_transform = self.t.y_transform
        # Min value needs to be transformed if e.g. objective function gets shifted.
        self.min_value = self.f_x_transform(jnp.array([[objective_func.min_y]]))
        # X: Input features, f_X: True objective values, Y: Training values (e.g. noisy)
        self.X, self.f_X, self.Y = self.sample_data(num_samples)

    def objective(self, X: Array) -> tuple[Array, Array, Array]:
        """
        The objective function to minimize. Automatically applies x and y transformations.
        Args:
            X: Array of shape (num_samples, input_dim).
        Returns:
            (X, f_X, Y): X:(num_samples, input_dim), f_X and Y: (num_samples, 1), where Y = noise_func(f(X)).
        """
        X_pre = self.x_transform_pre(X)
        f_X = self.f_x_transform(self._objective(X_pre))
        return self.x_transform_post(X), f_X, self.y_transform(f_X)

    def sample_data(self, num_samples) -> tuple[Array, Array, Array]:
        """
        Draws `num_samples` random samples from a sobol sequence and computes their objective values.
        Args:
            num_samples: Number of samples to draw.
        Returns:
            (X, f_X, Y): X.shape = (num_samples, input_dim), f_X.shape = Y.shape = (num_samples, 1).
        """
        self.key, subkey = random.split(self.key)
        np_key = np.random.default_rng(np.asarray(random.key_data(subkey)))
        # Draw samples
        sobol_engine = Sobol(d=self.input_dim, scramble=True, rng=np_key)
        sobol_samples = sobol_engine.random(n=num_samples)
        # Scale to bounds
        lb, ub = self.bounds
        X = jnp.array(lb + sobol_samples * (ub - lb))
        # Compute objective
        X, f_X, Y = self.objective(X)
        # Standardization
        Y = self.standardize_Y(Y)
        return X, f_X, Y

    def add_data_point(self, x: Array) -> None:
        """
        Add a sample point and its objective value to the dataset and standardize Y.
        Args:
            x: Array of shape (input_dim,).
        """
        x, f_x, _ = self.objective(x[None, :])
        self.X = jnp.vstack([self.X, x])
        self.f_X = jnp.vstack([self.f_X, f_x])
        self.Y = self.y_transform(self.f_X)
        # Standardization
        self.Y = self.standardize_Y(self.Y)
        assert self.X.shape[0] == self.f_X.shape[0] == self.Y.shape[0]

    def add_sampled_point(self, x: Array) -> None:
        """
        Add a sample point and its objective value to the dataset and standardize Y.
        Args:
            x: Array of shape (input_dim,).
        """
        x = self.x_transform_post.inverse(x)
        x, f_x, _ = self.objective(x[None, :])
        self.X = jnp.vstack([self.X, x])
        self.f_X = jnp.vstack([self.f_X, f_x])
        self.Y = self.y_transform(self.f_X)
        # Standardization
        self.Y = self.standardize_Y(self.Y)
        assert self.X.shape[0] == self.f_X.shape[0] == self.Y.shape[0]

    def standardize_Y(self, Y: Array) -> Array:
        """
        Standardizes the Y values and updates the mean and std attributes.
        Args:
            Y: Array of shape (num_samples, 1).
        Returns:
            Standardized Y array.
        """
        self.Y_mean = jnp.mean(Y)
        std = jnp.std(Y)
        self.Y_std = jnp.where(jnp.equal(std, 0), 1, std)
        Y = standardize(Y, self.Y_mean, self.Y_std)
        return Y

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        assert self.X.shape[0] == self.f_X.shape[0] == self.Y.shape[0]
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Get the sample at index `idx`.
        Args:
            idx: Index of the sample to retrieve.
        Returns:
            Tuple of (X[idx], Y[idx]).
        """
        return self.X[idx], self.Y[idx]

    @property
    def unstandardized_Y(self):
        """
        Returns the unstandardized Y values.
        """
        return self.unstandardize_func(self.Y)

    @property
    def unstandardize_func(self):
        """
        Returns a function to unstandardize Y values.
        """
        return partial(unstandardize, mean=self.Y_mean, std=self.Y_std)

    @property
    def config(self):
        """
        Returns the configuration of the dataset.
        """
        return {
            "objective_func": self.objective_func,
            "num_samples": self.init_num_samples,
            "t": self.t,
        }

    @classmethod
    def hydra_init(cls, **kwargs):
        """
        Hydra initialization method.
        """
        kwargs.pop("meta", None)
        return super().hydra_init(**kwargs)
