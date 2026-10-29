from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from tqdm import tqdm

from .classifier import PairwiseClassifier, average_classifiers
from .objectives import Objective


FloatArray = NDArray[np.float64]


class MetricsLogger(Protocol):
    def log_round(
        self, round_id: int, clients: list["Client"], diagnostics: dict[str, float]
    ) -> None: ...


@dataclass(frozen=True)
class IAFFBOConfig:
    rounds: int = 60
    initial_points: int = 50
    num_groups: int = 3
    server_sample_size: int = 50
    beta: float = 2.0
    acquisition: str = "ucb"
    classifier_epochs: int = 20
    classifier_learning_rate: float = 1e-3
    classifier_batch_size: int = 256
    classifier_weight_decay: float = 0.0
    max_pairwise_examples: int | None = None
    population_size: int = 100
    generations: int = 300
    phi: float = 0.1
    group_model_probability: float = 0.5
    gp_noise: float = 1e-6
    gp_restarts: int = 0
    kmeans_restarts: int = 10
    seed: int = 0
    # The last objective's client joins after this many rounds (None: all start together).
    warm_start_round: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Client:
    client_id: int
    objective: Objective
    x: FloatArray
    y: FloatArray
    classifier: PairwiseClassifier
    rng: np.random.Generator
    group_id: int = 0
    used_group_model: bool = False

    @property
    def best_x(self) -> FloatArray:
        return self.x[int(np.argmin(self.y))]

    @property
    def best_y(self) -> float:
        return float(np.min(self.y))


def latin_hypercube(size: int, dim: int, rng: np.random.Generator) -> FloatArray:
    samples = np.empty((size, dim), dtype=np.float64)
    for dimension in range(dim):
        samples[:, dimension] = (
            rng.permutation(size) + rng.random(size)
        ) / size
    return samples


def acquisition_scores(
    gp: GaussianProcessRegressor,
    candidates: FloatArray,
    *,
    kind: str,
    beta: float,
    best_value: float,
) -> FloatArray:
    """Return acquisition utilities to maximize for a minimized objective.

    The pairwise classifier and CSO always treat a larger score as better.
    Consequently, confidence bounds on the objective must be converted to
    utilities before constructing pairwise labels.  UCB is applied to the
    utility ``-f``; LCB is computed for ``f`` and then negated.  Both therefore
    reduce to ``-mean + beta * standard_deviation``.  Expected improvement is
    already a utility to maximize.
    """
    mean, standard_deviation = gp.predict(candidates, return_std=True)
    standard_deviation = np.maximum(standard_deviation, 1e-12)
    if kind == "ucb":
        # Upper confidence bound on the utility -f.
        return -mean + beta * standard_deviation
    if kind == "lcb":
        # Minimizing the lower confidence bound on f is equivalent to
        # maximizing its negative.
        return -(mean - beta * standard_deviation)
    if kind == "ei":
        improvement = best_value - mean
        z = improvement / standard_deviation
        return improvement * norm.cdf(z) + standard_deviation * norm.pdf(z)
    raise ValueError("acquisition must be one of: ucb, lcb, ei.")


def pairwise_dataset(
    points: FloatArray,
    scores: FloatArray,
    rng: np.random.Generator,
    max_examples: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Build both orders of every unordered comparison, including ties."""
    first, second = np.triu_indices(len(points), k=1)
    x_forward = np.hstack((points[first], points[second]))
    x_reverse = np.hstack((points[second], points[first]))
    pair_inputs = np.vstack((x_forward, x_reverse))
    pair_labels = np.concatenate(
        (
            (scores[first] > scores[second]).astype(np.float64),
            (scores[second] > scores[first]).astype(np.float64),
        )
    )
    if max_examples is not None and len(pair_inputs) > max_examples:
        # Subsample unordered pairs, then retain both directions so each
        # comparison is represented consistently, including tied comparisons.
        selected_pairs = rng.choice(
            len(first), size=max_examples // 2, replace=False
        )
        selected = np.concatenate((selected_pairs, selected_pairs + len(first)))
        pair_inputs, pair_labels = pair_inputs[selected], pair_labels[selected]
    return pair_inputs, pair_labels


def competitive_swarm_optimize(
    classifier: PairwiseClassifier,
    dim: int,
    *,
    population_size: int,
    generations: int,
    phi: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Optimize an implicit acquisition function using modified CSO."""
    positions = rng.random((population_size, dim))
    velocities = np.zeros_like(positions)
    for _ in range(generations):
        mean_position = positions.mean(axis=0)
        order = rng.permutation(population_size)
        # With an odd population, one particle receives a bye this generation.
        for pair_start in range(0, population_size - 1, 2):
            first_id, second_id = order[pair_start : pair_start + 2]
            first_wins = bool(
                classifier.prefers(
                    positions[first_id : first_id + 1],
                    positions[second_id : second_id + 1],
                )[0]
            )
            winner_id, loser_id = (
                (first_id, second_id) if first_wins else (second_id, first_id)
            )
            r1, r2, r3 = rng.random(3)
            velocities[loser_id] = (
                r1 * velocities[loser_id]
                + phi * r2 * (positions[winner_id] - positions[loser_id])
                + r3 * (mean_position - positions[loser_id])
            )
            positions[loser_id] = np.clip(
                positions[loser_id] + velocities[loser_id], 0.0, 1.0
            )

    # A shuffled knockout tournament defines a champion even for a non-transitive model.
    contenders = list(rng.permutation(population_size))
    while len(contenders) > 1:
        next_round: list[int] = []
        for index in range(0, len(contenders) - 1, 2):
            first_id, second_id = contenders[index : index + 2]
            first_wins = bool(
                classifier.prefers(
                    positions[first_id : first_id + 1],
                    positions[second_id : second_id + 1],
                )[0]
            )
            next_round.append(first_id if first_wins else second_id)
        if len(contenders) % 2:
            next_round.append(contenders[-1])
        contenders = next_round
    return positions[contenders[0]].copy()


class IAFFBO:
    def __init__(self, objectives: list[Objective], config: IAFFBOConfig):
        if not objectives:
            raise ValueError("At least one client objective is required.")
        if len({objective.dim for objective in objectives}) != 1:
            raise ValueError("All clients must have the same input dimension.")
        if config.rounds < 0 or config.generations < 0 or config.classifier_epochs < 0:
            raise ValueError("rounds, generations, and classifier_epochs must be non-negative.")
        if config.initial_points < 1 or config.server_sample_size < 1:
            raise ValueError("initial_points and server_sample_size must be positive.")
        if config.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if config.num_groups < 1:
            raise ValueError("num_groups must be positive.")
        if not 0.0 <= config.group_model_probability <= 1.0:
            raise ValueError("group_model_probability must be in [0, 1].")
        if config.phi < 0.0 or config.beta < 0.0 or config.gp_noise <= 0.0:
            raise ValueError("phi and beta must be non-negative; gp_noise must be positive.")
        if config.classifier_learning_rate <= 0.0 or config.classifier_batch_size < 1:
            raise ValueError("Classifier learning rate and batch size must be positive.")
        if config.max_pairwise_examples is not None and config.max_pairwise_examples < 2:
            raise ValueError("max_pairwise_examples must be at least 2.")
        if config.gp_restarts < 0 or config.kmeans_restarts < 1:
            raise ValueError("gp_restarts must be non-negative and kmeans_restarts positive.")
        if config.acquisition not in {"ucb", "lcb", "ei"}:
            raise ValueError("acquisition must be one of: ucb, lcb, ei.")
        if config.warm_start_round is not None:
            if not 0 < config.warm_start_round < config.rounds:
                raise ValueError("warm_start_round must lie strictly between 0 and rounds.")
            if len(objectives) < 2:
                raise ValueError("Warm starting requires at least two client objectives.")

        self.objectives = objectives
        self.config = config
        self.dim = objectives[0].dim
        self.rng = np.random.default_rng(config.seed)
        self.clients = self._initialize_clients()
        # The warm-started client draws its initial design with the others but
        # only takes part in training, aggregation, and acquisition after joining.
        self.pending_clients: list[Client] = []
        if config.warm_start_round is not None:
            self.pending_clients.append(self.clients.pop())
        self.server_samples = self.rng.random((config.server_sample_size, self.dim))
        self.group_models: dict[int, PairwiseClassifier] = {}
        self.last_diagnostics: dict[str, float] = {}

    def _initialize_clients(self) -> list[Client]:
        clients = []
        child_seeds = np.random.SeedSequence(self.config.seed).spawn(
            len(self.objectives)
        )
        for client_id, (objective, seed) in enumerate(
            zip(self.objectives, child_seeds)
        ):
            rng = np.random.default_rng(seed)
            x = latin_hypercube(self.config.initial_points, self.dim, rng)
            y = np.asarray([objective(point) for point in x], dtype=np.float64)
            clients.append(
                Client(
                    client_id,
                    objective,
                    x,
                    y,
                    PairwiseClassifier.random(self.dim, rng),
                    rng,
                )
            )
        return clients

    def _fit_gp(self, client: Client) -> GaussianProcessRegressor:
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            1.0, (1e-3, 1e3)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config.gp_noise,
            normalize_y=False,
            n_restarts_optimizer=self.config.gp_restarts,
            random_state=self.config.seed + client.client_id,
        )
        with warnings.catch_warnings():
            # Boundary optima are common for tiny early-round datasets and do
            # not invalidate the fitted posterior.
            warnings.simplefilter("ignore", ConvergenceWarning)
            return gp.fit(client.x, client.y)

    def _train_client(self, client: Client) -> float:
        if self.group_models:
            client.classifier = self.group_models[client.group_id].copy()
        gp = self._fit_gp(client)
        points = np.vstack((client.x, self.server_samples))
        scores = acquisition_scores(
            gp,
            points,
            kind=self.config.acquisition,
            beta=self.config.beta,
            best_value=client.best_y,
        )
        pair_inputs, pair_labels = pairwise_dataset(
            points,
            scores,
            client.rng,
            self.config.max_pairwise_examples,
        )
        if len(pair_inputs) == 0:
            return float("nan")
        losses = client.classifier.fit(
            pair_inputs,
            pair_labels,
            epochs=self.config.classifier_epochs,
            learning_rate=self.config.classifier_learning_rate,
            batch_size=self.config.classifier_batch_size,
            rng=client.rng,
            weight_decay=self.config.classifier_weight_decay,
        )
        return losses[-1] if losses else float("nan")

    def _cluster_and_aggregate(self, round_id: int) -> None:
        vectors = np.vstack(
            [client.classifier.to_vector() for client in self.clients]
        )
        group_count = min(self.config.num_groups, len(self.clients))
        if group_count == 1:
            labels = np.zeros(len(self.clients), dtype=int)
        else:
            with warnings.catch_warnings():
                # Identical client vectors can make fewer than G distinct
                # clusters; in that case aggregate the non-empty groups.
                warnings.simplefilter("ignore", ConvergenceWarning)
                labels = KMeans(
                    n_clusters=group_count,
                    n_init=self.config.kmeans_restarts,
                    random_state=self.config.seed + round_id,
                ).fit_predict(vectors)
        self.group_models = {}
        for group_id, original_label in enumerate(np.unique(labels)):
            members = [
                client
                for client, label in zip(self.clients, labels)
                if label == original_label
            ]
            for client in members:
                client.group_id = group_id
            self.group_models[group_id] = average_classifiers(
                [client.classifier for client in members], self.dim
            )

    def run(self, logger: MetricsLogger | None = None) -> list[Client]:
        for round_id in tqdm(range(self.config.rounds), desc="Rounds"):
            losses = [self._train_client(client) for client in self.clients]
            self._cluster_and_aggregate(round_id)

            for client in self.clients:
                use_group = (
                    client.rng.random() < self.config.group_model_probability
                )
                client.used_group_model = use_group
                search_model = (
                    self.group_models[client.group_id]
                    if use_group
                    else client.classifier
                )
                proposal = competitive_swarm_optimize(
                    search_model,
                    self.dim,
                    population_size=self.config.population_size,
                    generations=self.config.generations,
                    phi=self.config.phi,
                    rng=client.rng,
                )
                client.x = np.vstack((client.x, proposal))
                client.y = np.append(client.y, client.objective(proposal))

            self.server_samples = self.rng.random(
                (self.config.server_sample_size, self.dim)
            )
            finite_losses = np.asarray(losses)[np.isfinite(losses)]
            diagnostics = {
                "server/num_groups": float(len(self.group_models)),
                "server/group_model_fraction": float(
                    np.mean([client.used_group_model for client in self.clients])
                ),
                "classifier/mean_loss": (
                    float(finite_losses.mean()) if len(finite_losses) else float("nan")
                ),
            }
            self.last_diagnostics = diagnostics
            if logger is not None:
                logger.log_round(round_id, self.clients, diagnostics)
            if round_id + 1 == self.config.warm_start_round:
                self.clients.extend(self.pending_clients)
                self.pending_clients = []
        return self.clients
