import jax
from jax import random
from jaxtyping import Array, Key

from fedbo.data.objective_dataset import ObjectiveDataset
from fedbo.transformations import ObjectiveTransform, minmax_scaling
from fedbo.utils.minimize import minimize_func
from fedbo.data.objectives import (
    BaseObjective,
    ObjectiveWrapper,
    AckleyObjective,
    CosineObjective,
    RastriginObjective,
    GriewankObjective,
    SphereObjective,
)


class CombinationDataset(ObjectiveDataset):
    def __init__(
        self,
        num_samples: int,
        input_dim: int,
        key: Key,
        t: ObjectiveTransform = None,
    ):
        """
        Intialize Combination Dataset.
        Args:
            num_samples: The initial number of samples.
            input_dim: The dimension of the input features.
            key: A PRNG Key.
            t: Optional transformations for the objective function.
        """
        bounds = self.bounds()
        self.objectives = [
            AckleyObjective(input_dim),
            CosineObjective(input_dim),
            RastriginObjective(input_dim),
            GriewankObjective(input_dim),
            SphereObjective(input_dim),
        ]
        num_objectives = len(self.objectives)
        self.shifts = [0, -1, 8, -4, 3]
        # Generate random weights and normalize them to sum to 1
        key, subkey = random.split(key)
        weights = random.uniform(subkey, shape=(num_objectives,))
        self.weights = weights / weights.sum()

        @jax.jit
        def objective_func(X: Array):
            return sum(
                w * scaled_objective(obj, X - shift) for w, obj, shift in zip(self.weights, self.objectives, self.shifts)
            )

        key, subkey = random.split(key)
        _, min_y = minimize_func(objective_func, input_dim, bounds, subkey, 1000 * input_dim)
        key, subkey = random.split(key)
        _, max_y = minimize_func(lambda x: -objective_func(x), input_dim, bounds, subkey)

        objective_wrapper = ObjectiveWrapper(objective_func, input_dim, bounds, min_y.squeeze(), -max_y.squeeze())
        super().__init__(objective_wrapper, num_samples, key, t)

    @staticmethod
    def bounds():
        """
        Returns the bounds of the input features for hydra configuration.
        """
        return (-5, 10)

    @property
    def config(self):
        """
        Returns the configuration of the dataset.
        """
        return {
            "num_samples": self.init_num_samples,
            "input_dim": self.input_dim,
            "t": self.t,
        }


def scaled_objective(f: BaseObjective, X: Array):
    """
    Scales the objective function values to the range [0, 1].
    Args:
        f: The objective function.
        X: Input array.
    Returns:
        Scaled objective values.
    """
    return minmax_scaling(f(X), f.min_y, f.max_y)
