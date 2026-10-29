from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_2d(x: ArrayLike) -> FloatArray:
    x_array = np.asarray(x, dtype=np.float64)
    return x_array[None, :] if x_array.ndim == 1 else x_array


def _kmeans(
    x: FloatArray,
    num_centers: int,
    rng: np.random.Generator,
    max_iterations: int,
) -> FloatArray:
    """Small NumPy k-means implementation which also supports k > n."""
    replace = len(x) < num_centers
    centers = x[rng.choice(len(x), size=num_centers, replace=replace)].copy()
    for _ in range(max_iterations):
        distances = np.maximum(
            np.sum(x**2, axis=1)[:, None]
            + np.sum(centers**2, axis=1)[None, :]
            - 2.0 * x @ centers.T,
            0.0,
        )
        labels = np.argmin(distances, axis=1)
        updated = centers.copy()
        for center_id in range(num_centers):
            members = x[labels == center_id]
            updated[center_id] = members.mean(axis=0) if len(members) else x[rng.integers(len(x))]
        if np.allclose(updated, centers, rtol=1e-6, atol=1e-8):
            break
        centers = updated
    return centers


def _center_widths(centers: FloatArray) -> FloatArray:
    distances = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2))
    distances[distances == 0.0] = np.inf
    nearest = distances.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    fallback = float(np.median(finite)) if len(finite) else 1.0
    return np.maximum(np.where(np.isfinite(nearest), nearest, fallback), 1e-4)


def _center_metric_order(centers: FloatArray) -> NDArray[np.intp]:
    """Return the sorted-averaging order from equation (11)."""
    matching_metric = np.sum(centers**2, axis=1)
    return np.argsort(matching_metric, kind="stable")


@dataclass(frozen=True)
class RBFNetwork:
    """Gaussian RBF network used by FD-EMD.

    Centers and widths define the RBF features and the output weights are
    trained with least squares initially. Subsequent client and distillation
    updates recalculate the centers with k-means before optimizing the L2 loss.
    """

    centers: FloatArray
    widths: FloatArray
    weights: FloatArray

    @property
    def input_dim(self) -> int:
        return self.centers.shape[1]

    @property
    def num_centers(self) -> int:
        return self.centers.shape[0]

    def _features(self, x: ArrayLike) -> FloatArray:
        x_array = _as_2d(x)
        widths = np.maximum(np.abs(self.widths), 1e-8)
        squared_distance = np.maximum(
            np.sum(x_array**2, axis=1)[:, None]
            + np.sum(self.centers**2, axis=1)[None, :]
            - 2.0 * x_array @ self.centers.T,
            0.0,
        )
        return np.exp(-squared_distance / (2.0 * widths[None, :] ** 2))

    def predict(self, x: ArrayLike) -> FloatArray:
        return self._features(x) @ self.weights

    def to_vector(self) -> FloatArray:
        """Serialize raw RBF parameters for the Gaussian in paper equation (7)."""
        return np.concatenate((self.centers.ravel(), self.widths, self.weights))

    @classmethod
    def from_vector(cls, vector: ArrayLike, input_dim: int, num_centers: int) -> "RBFNetwork":
        vector_array = np.asarray(vector, dtype=np.float64)
        expected = num_centers * input_dim + 2 * num_centers
        if vector_array.size != expected:
            raise ValueError(f"Expected {expected} RBF parameters, got {vector_array.size}.")
        offset = num_centers * input_dim
        centers = vector_array[:offset].reshape(num_centers, input_dim)
        widths = np.maximum(np.abs(vector_array[offset : offset + num_centers]), 1e-8)
        offset += num_centers
        weights = vector_array[offset : offset + num_centers]
        return cls(centers, widths, weights)

    @classmethod
    def random(
        cls,
        input_dim: int,
        num_centers: int,
        rng: np.random.Generator,
    ) -> "RBFNetwork":
        """Initialize the paper's pre-round global model in the unit search space."""
        centers = rng.random((num_centers, input_dim))
        centers = centers[_center_metric_order(centers)]
        widths = _center_widths(centers)
        weights = rng.normal(0.0, 1.0 / np.sqrt(num_centers), size=num_centers)
        return cls(centers, widths, weights)

    def sorted_by_center_metric(self) -> "RBFNetwork":
        """Sort nodes by M_j = sum_i(c_{j,i}^2) for sorted averaging. According to: J. Xu, Y. Jin, W. Du, and S. Gu, “A federated data-driven evolutionary
algorithm,” Knowl.-Based Syst., vol. 233, Dec. 2021, Art. no. 107532.
[Online]. Available: https://www.sciencedirect.com/science/article/pii/
S0950705121007942"""
        order = _center_metric_order(self.centers)
        return RBFNetwork(
            self.centers[order].copy(),
            self.widths[order].copy(),
            self.weights[order].copy(),
        )

    def train_l2(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        epochs: int,
        learning_rate: float,
        rng: np.random.Generator,
        kmeans_iterations: int = 100,
    ) -> "RBFNetwork":
        """Recalculate RBF centers, then run full-batch L2 updates."""
        x_array = _as_2d(x)
        y_array = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(x_array) != len(y_array) or not len(x_array):
            raise ValueError("x and y must contain the same non-zero number of observations.")
        if epochs < 0:
            raise ValueError("epochs must be non-negative.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")

        centers = _kmeans(x_array, self.num_centers, rng, kmeans_iterations)
        centers = centers[_center_metric_order(centers)]
        widths = _center_widths(centers)
        updated_model = RBFNetwork(
            centers,
            widths,
            self.sorted_by_center_metric().weights.copy(),
        )
        features = updated_model._features(x_array)
        weights = updated_model.weights.copy()
        for _ in range(epochs):
            residual = features @ weights - y_array
            gradient = (2.0 / len(x_array)) * features.T @ residual
            weights -= learning_rate * gradient
        return RBFNetwork(centers, widths, weights)

    @classmethod
    def fit(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        num_centers: int,
        rng: np.random.Generator,
        ridge: float = 0.0,
        kmeans_iterations: int = 100,
    ) -> "RBFNetwork":
        x_array = _as_2d(x)
        y_array = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(x_array) != len(y_array) or not len(x_array):
            raise ValueError("x and y must contain the same non-zero number of observations.")
        if num_centers < 1:
            raise ValueError("num_centers must be positive.")

        centers = _kmeans(x_array, num_centers, rng, kmeans_iterations)
        # Apply the sorted-averaging matching metric so corresponding parameter
        # blocks can be aggregated by node index.
        centers = centers[_center_metric_order(centers)]
        widths = _center_widths(centers)
        squared_distance = np.maximum(
            np.sum(x_array**2, axis=1)[:, None]
            + np.sum(centers**2, axis=1)[None, :]
            - 2.0 * x_array @ centers.T,
            0.0,
        )
        features = np.exp(-squared_distance / (2.0 * widths[None, :] ** 2))
        if ridge > 0.0:
            gram = features.T @ features
            weights = np.linalg.solve(
                gram + ridge * np.eye(num_centers),
                features.T @ y_array,
            )
        else:
            # pinv implements equation (3) robustly when h > K.
            weights = np.linalg.pinv(features) @ y_array
        return cls(centers, widths, weights)
