from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset


FloatArray = NDArray[np.float64]


class PairwiseClassifier(nn.Module):
    """Small ReLU MLP whose parameters can be flattened and averaged."""

    def __init__(self, dim: int):
        super().__init__()
        hidden = (max(1, round(1.5 * dim)), dim, max(1, round(0.5 * dim)))
        sizes = (2 * dim, *hidden, 1)
        modules: list[nn.Module] = []
        for layer, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            modules.append(nn.Linear(fan_in, fan_out))
            if layer < len(sizes) - 2:
                modules.append(nn.ReLU())
        self.network = nn.Sequential(*modules).double()

    @classmethod
    def random(cls, dim: int, rng: np.random.Generator) -> "PairwiseClassifier":
        # Derive PyTorch's initialization seed from the client's NumPy
        # generator without changing PyTorch's process-wide RNG state.
        seed = int(rng.integers(0, np.iinfo(np.int64).max))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            classifier = cls(dim)
            for module in classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    nn.init.zeros_(module.bias)
        return classifier

    @property
    def input_dim(self) -> int:
        first_layer = self.network[0]
        assert isinstance(first_layer, nn.Linear)
        return first_layer.in_features

    def forward(self, pair_inputs: Tensor) -> Tensor:
        return self.network(pair_inputs).squeeze(-1)

    def copy(self) -> "PairwiseClassifier":
        return PairwiseClassifier.from_vector(self.to_vector(), self.input_dim // 2)

    def logits(self, pair_inputs: FloatArray) -> FloatArray:
        inputs = torch.as_tensor(pair_inputs, dtype=torch.float64)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self(inputs).cpu().numpy().copy()
        self.train(was_training)
        return logits

    def predict_proba(self, pair_inputs: FloatArray) -> FloatArray:
        inputs = torch.as_tensor(pair_inputs, dtype=torch.float64)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self(inputs)).cpu().numpy().copy()
        self.train(was_training)
        return probabilities

    def prefers(self, first: FloatArray, second: FloatArray) -> NDArray[np.bool_]:
        first_array = np.atleast_2d(np.asarray(first, dtype=np.float64))
        second_array = np.atleast_2d(np.asarray(second, dtype=np.float64))
        if len(first_array) != len(second_array):
            raise ValueError("first and second must contain the same number of points.")
        return self.predict_proba(np.hstack((first_array, second_array))) >= 0.5

    def to_vector(self) -> FloatArray:
        return (
            parameters_to_vector(self.parameters())
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
        )

    @classmethod
    def from_vector(cls, vector: FloatArray, dim: int) -> "PairwiseClassifier":
        classifier = cls(dim)
        flat = np.asarray(vector, dtype=np.float64).reshape(-1)
        expected = sum(parameter.numel() for parameter in classifier.parameters())
        if flat.size != expected:
            raise ValueError(f"Expected {expected} parameters, received {flat.size}.")
        vector_to_parameters(
            torch.tensor(flat, dtype=torch.float64),
            classifier.parameters(),
        )
        return classifier

    def fit(
        self,
        pair_inputs: FloatArray,
        labels: FloatArray,
        *,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        rng: np.random.Generator,
        weight_decay: float = 0.0,
    ) -> list[float]:
        """Train with PyTorch mini-batch Adam and binary cross entropy."""
        x = np.asarray(pair_inputs, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.input_dim or len(x) != len(y):
            raise ValueError("Invalid pairwise training data shape.")
        if len(x) == 0:
            raise ValueError("At least one pairwise example is required.")

        dataset = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
        )
        generator = torch.Generator().manual_seed(
            int(rng.integers(0, np.iinfo(np.int64).max))
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        losses: list[float] = []

        self.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            losses.append(epoch_loss / len(dataset))
        return losses


def average_classifiers(
    classifiers: list[PairwiseClassifier],
    dim: int,
) -> PairwiseClassifier:
    if not classifiers:
        raise ValueError("At least one classifier is required.")
    return PairwiseClassifier.from_vector(
        np.mean([classifier.to_vector() for classifier in classifiers], axis=0),
        dim,
    )
