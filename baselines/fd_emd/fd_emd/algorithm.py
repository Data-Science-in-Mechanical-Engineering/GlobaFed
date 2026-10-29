from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .objectives import Objective
from .rbfn import RBFNetwork


FloatArray = NDArray[np.float64]


class MetricsLogger(Protocol):
    def log_round(self, round_id: int, clients: list["Client"], diagnostics: dict[str, float]) -> None: ...


@dataclass(frozen=True)
class FDEMDConfig:
    rounds: int = 60
    initial_points: int = 50
    num_centers: int | None = None
    ensemble_size: int = 20
    auxiliary_size: int = 1000
    population_size: int = 50
    generations: int = 20
    lcb_weight: float = 2.0
    mutation_probability: float | None = None
    mutation_scale: float = 0.1
    local_epochs: int = 5
    learning_rate: float = 0.06
    ridge: float = 0.0
    kmeans_iterations: int = 100
    seed: int = 0
    # The last objective's client joins after this many rounds (None: all start together).
    warm_start_round: int | None = None

    def resolved_num_centers(self, dim: int) -> int:
        return 2 * dim + 1 if self.num_centers is None else self.num_centers

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Client:
    client_id: int
    objective: Objective
    x: FloatArray
    y: FloatArray
    model: RBFNetwork
    rng: np.random.Generator


def latin_hypercube(size: int, dim: int, rng: np.random.Generator) -> FloatArray:
    samples = np.empty((size, dim), dtype=np.float64)
    for dimension in range(dim):
        samples[:, dimension] = (rng.permutation(size) + rng.random(size)) / size
    return samples


def _lcb(local: RBFNetwork, global_model: RBFNetwork, candidates: FloatArray, weight: float) -> FloatArray:
    predictions = np.vstack((local.predict(candidates), global_model.predict(candidates)))
    return predictions.mean(axis=0) - weight * predictions.std(axis=0)


def evolve_candidate(
    local: RBFNetwork,
    global_model: RBFNetwork,
    dim: int,
    config: FDEMDConfig,
    rng: np.random.Generator,
) -> tuple[FloatArray, float]:
    population = rng.random((config.population_size, dim))
    mutation_probability = (
        1.0 / dim if config.mutation_probability is None else config.mutation_probability
    )
    for _ in range(config.generations):
        scores = _lcb(local, global_model, population, config.lcb_weight)
        elite_count = max(2, config.population_size // 5)
        elites = population[np.argsort(scores)[:elite_count]]
        parent_ids = rng.integers(elite_count, size=(config.population_size - elite_count, 2))
        alpha = rng.random((config.population_size - elite_count, 1))
        offspring = alpha * elites[parent_ids[:, 0]] + (1.0 - alpha) * elites[parent_ids[:, 1]]
        mutation_mask = rng.random(offspring.shape) < mutation_probability
        offspring += mutation_mask * rng.normal(0.0, config.mutation_scale, offspring.shape)
        population = np.vstack((elites, np.clip(offspring, 0.0, 1.0)))
    scores = _lcb(local, global_model, population, config.lcb_weight)
    best = int(np.argmin(scores))
    return population[best], float(scores[best])


class FDEMD:
    def __init__(self, objectives: list[Objective], config: FDEMDConfig):
        if not objectives:
            raise ValueError("At least one client objective is required.")
        if len({objective.dim for objective in objectives}) != 1:
            raise ValueError("All clients must have the same input dimension.")
        if config.rounds < 0 or config.generations < 0 or config.local_epochs < 0:
            raise ValueError("rounds, generations, and local_epochs must be non-negative.")
        if config.initial_points < 1 or config.ensemble_size < 1 or config.auxiliary_size < 1:
            raise ValueError("initial_points, ensemble_size, and auxiliary_size must be positive.")
        if config.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if config.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if config.mutation_probability is not None and not 0.0 <= config.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be in [0, 1].")
        if config.warm_start_round is not None:
            if not 0 < config.warm_start_round < config.rounds:
                raise ValueError("warm_start_round must lie strictly between 0 and rounds.")
            if len(objectives) < 2:
                raise ValueError("Warm starting requires at least two client objectives.")
        self.objectives = objectives
        self.config = config
        self.dim = objectives[0].dim
        self.num_centers = config.resolved_num_centers(self.dim)
        self.rng = np.random.default_rng(config.seed)
        self.clients = self._initialize_clients()
        # The warm-started client draws its initial design with the others but
        # only takes part in training, aggregation, and acquisition after joining.
        self.pending_clients: list[Client] = []
        if config.warm_start_round is not None:
            self.pending_clients.append(self.clients.pop())
        # Algorithms 1 and 3 initialize w_0^g before the first client round.
        self.global_model = RBFNetwork.random(
            self.dim,
            self.num_centers,
            self.rng,
        )
        self.last_diagnostics: dict[str, float] = {}

    def _fit(self, x: FloatArray, y: FloatArray, rng: np.random.Generator) -> RBFNetwork:
        return RBFNetwork.fit(
            x,
            y,
            self.num_centers,
            rng,
            ridge=self.config.ridge,
            kmeans_iterations=self.config.kmeans_iterations,
        )

    def _initialize_clients(self) -> list[Client]:
        clients = []
        seed_sequence = np.random.SeedSequence(self.config.seed)
        child_seeds = seed_sequence.spawn(len(self.objectives))
        for client_id, (objective, seed) in enumerate(zip(self.objectives, child_seeds)):
            rng = np.random.default_rng(seed)
            x = latin_hypercube(self.config.initial_points, self.dim, rng)
            y = np.array([objective(point) for point in x], dtype=np.float64)
            clients.append(Client(client_id, objective, x, y, self._fit(x, y, rng), rng))
        return clients

    def _distill_global(self) -> tuple[RBFNetwork, dict[str, float]]:
        sorted_models = [
            client.model.sorted_by_center_metric()
            for client in self.clients
        ]
        vectors = np.vstack([model.to_vector() for model in sorted_models])
        mean = vectors.mean(axis=0)
        variance = np.mean((vectors - mean) ** 2, axis=0)
        sampled_vectors = self.rng.normal(
            loc=mean,
            scale=np.sqrt(variance),
            size=(self.config.ensemble_size, mean.size),
        )
        ensemble = [
            RBFNetwork.from_vector(vector, self.dim, self.num_centers)
            for vector in sampled_vectors
        ]
        auxiliary_x = self.rng.random((self.config.auxiliary_size, self.dim))
        pseudo_y = np.zeros(self.config.auxiliary_size, dtype=np.float64)
        for model in ensemble:
            pseudo_y += model.predict(auxiliary_x)
        pseudo_y /= self.config.ensemble_size
        student = self.global_model.train_l2(
            auxiliary_x,
            pseudo_y,
            epochs=self.config.local_epochs,
            learning_rate=self.config.learning_rate,
            rng=self.rng,
            kmeans_iterations=self.config.kmeans_iterations,
        )
        diagnostics = {
            "server/parameter_variance_mean": float(variance.mean()),
            "server/pseudo_label_mean": float(pseudo_y.mean()),
            "server/pseudo_label_std": float(pseudo_y.std()),
        }
        return student, diagnostics

    def run(self, logger: MetricsLogger | None = None) -> list[Client]:
        for round_id in tqdm(range(self.config.rounds), desc="Rounds"):
            lcb_values = []
            for client in self.clients:
                x_next, lcb = evolve_candidate(
                    client.model,
                    self.global_model,
                    self.dim,
                    self.config,
                    client.rng,
                )
                y_next = client.objective(x_next)
                client.x = np.vstack((client.x, x_next))
                client.y = np.append(client.y, y_next)
                client.model = client.model.train_l2(
                    client.x,
                    client.y,
                    epochs=self.config.local_epochs,
                    learning_rate=self.config.learning_rate,
                    rng=client.rng,
                    kmeans_iterations=self.config.kmeans_iterations,
                )
                lcb_values.append(lcb)

            self.global_model, diagnostics = self._distill_global()
            self.last_diagnostics = diagnostics
            diagnostics["acquisition/mean_lcb"] = float(np.mean(lcb_values))
            if logger is not None:
                logger.log_round(round_id, self.clients, diagnostics)
            if round_id + 1 == self.config.warm_start_round:
                self.clients.extend(self.pending_clients)
                self.pending_clients = []
        return self.clients
