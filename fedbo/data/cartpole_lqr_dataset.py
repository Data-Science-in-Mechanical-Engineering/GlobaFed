import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Key

from fedbo.data.objective_dataset import ObjectiveDataset
from fedbo.transformations import ObjectiveTransform
from fedbo.data.objectives import CartpoleLQRObjective


class CartpoleLQRDataset(ObjectiveDataset):
    def __init__(
        self,
        num_samples: int,
        key: Key,
        t: ObjectiveTransform = None,
    ):
        """
        Intialize Cartpole LQR Dataset.
        Args:
            num_samples: The initial number of samples.
            key: A PRNG Key.
            t: Optional transformations for the objective function.
        """
        self.key, subkey = random.split(key)
        objective_func = CartpoleLQRObjective(subkey)

        self.init_num_samples = num_samples
        self.objective_func = objective_func
        self._objective = objective_func
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

    @property
    def config(self):
        """
        Returns the configuration of the dataset.
        """
        return {
            "num_samples": self.init_num_samples,
            "t": self.t,
        }
