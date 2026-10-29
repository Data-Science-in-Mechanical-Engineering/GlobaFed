from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize


FloatArray = NDArray[np.float64]
ObjectiveFunction = Callable[[FloatArray], float]


@dataclass(frozen=True)
class Objective:
    name: str
    dim: int
    lower: FloatArray
    upper: FloatArray
    optimum: float
    function: ObjectiveFunction

    def __call__(self, x: FloatArray, *args, **kwargs) -> float:
        x_original = self.lower + np.asarray(x, dtype=np.float64) * (
            self.upper - self.lower
        )
        return float(self.function(x_original))

    @property
    def d(self) -> int:
        """Compatibility alias used by FMTBO's client implementation."""
        return self.dim

    @property
    def x_lb(self) -> float:
        """Lower bound in the normalized search space used by all baselines."""
        return 0.0

    @property
    def x_ub(self) -> float:
        """Upper bound in the normalized search space used by all baselines."""
        return 1.0

    @property
    def x_bound(self) -> list[float]:
        return [self.x_lb, self.x_ub]

    @property
    def best_y(self) -> float:
        """Compatibility alias for legacy baseline regret reporting."""
        return self.optimum

    def init_x_with_bound(
        self,
        size: int,
        iid: bool,
        pb: tuple[float, float] | None = None,
    ) -> FloatArray:
        bounds = (self.x_lb, self.x_ub) if iid or pb is None else pb
        return np.random.uniform(*bounds, size=(size, self.dim))

    def random_uniform_x(
        self,
        size: int,
        iid: bool,
        pb: tuple[float, float] | None = None,
    ) -> FloatArray:
        return self.init_x_with_bound(size, iid, pb)


def _ackley(x: FloatArray) -> float:
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        - np.exp(np.mean(np.cos(2.0 * np.pi * x)))
        + 20.0
        + np.e
    )


def _rastrigin(x: FloatArray) -> float:
    return float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0))


def _sphere(x: FloatArray) -> float:
    return float(x @ x)


def _cosine(x: FloatArray) -> float:
    return float(np.sum(np.cos(x) * ((0.1 / (2.0 * np.pi)) * np.abs(x) - 1.0)))


def _griewank(x: FloatArray) -> float:
    indices = np.arange(1, len(x) + 1)
    return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(indices))) + 1.0)


def _rosenbrock(x: FloatArray) -> float:
    return float(np.sum(100.0 * (x[1:]** 2 - x[:-1]) ** 2 + (x[:-1] - 1.0) ** 2))


def _schwefel(x: FloatArray) -> float:
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def _weierstrass(x: FloatArray) -> float:
    k = np.arange(22)
    a, b = 0.5, 3.0
    # The expanded form keeps the implementation vectorized and has f(0) = 0.
    terms = (a**k)[None, :] * np.cos(2.0 * np.pi * (b**k)[None, :] * (x[:, None] + 0.5))
    correction = len(x) * np.sum(a**k * np.cos(np.pi * b**k))
    return float(np.sum(terms) - correction)


def _branin(x: FloatArray) -> float:
    x1, x2 = x
    return float(
        (x2 - (5.1 / (4.0 * np.pi**2)) * x1**2 + (5.0 / np.pi) * x1 - 6.0) ** 2
        + 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * np.cos(x1)
        + 10.0
    )


def _hartmann6d(x: FloatArray) -> float:
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    coefficients = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    locations = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    return float(-np.sum(alpha * np.exp(-np.sum(coefficients * (x - locations) ** 2, axis=1))))


_FUNCTIONS: dict[str, tuple[ObjectiveFunction, float, float]] = {
    "ackley": (_ackley, -5.0, 10.0),
    "rastrigin": (_rastrigin, -5.12, 5.12),
    "sphere": (_sphere, -5.0, 5.0),
    "griewank": (_griewank, -100.0, 100.0),
    "rosenbrock": (_rosenbrock, -5.0, 10.0),
    "schwefel": (_schwefel, -500.0, 500.0),
}

TASK_NAMES = (
    "manytask",
    "combination",
    "ackley_shifted",
    "ackley",
    "branin_shifted",
    "hartmann6d_shifted",
    "cartpole",
    "rastrigin",
    "sphere",
    "griewank",
    "rosenbrock",
    "schwefel",
)


def _shifted_objectives(
    name: str,
    dim: int,
    num_clients: int,
    rng: np.random.Generator,
) -> list[Objective]:
    specs: dict[str, tuple[ObjectiveFunction, int | None, float, float, float, float, float]] = {
        "ackley_shifted": (_ackley, None, -5.0, 10.0, -3.0, 3.0, 0.0),
        "branin_shifted": (_branin, 2, 0.0, 10.0, -3.0, 3.0, 0.397887),
        "hartmann6d_shifted": (_hartmann6d, 6, 0.0, 1.0, -0.3, 0.15, -3.32237),
    }
    base_function, required_dim, lower, upper, shift_lower, shift_upper, base_optimum = specs[name]
    if required_dim is not None and dim != required_dim:
        raise ValueError(f"{name} requires dim={required_dim}, received dim={dim}.")

    objectives = []
    for client_id in range(num_clients):
        # benchmark.py deliberately uses one scalar input shift per client.
        x_shift = float(rng.uniform(shift_lower, shift_upper))
        y_shift = float(rng.uniform(-3.0, 3.0))

        def shifted(
            x: FloatArray,
            x_shift: float = x_shift,
            y_shift: float = y_shift,
            base_function: ObjectiveFunction = base_function,
        ) -> float:
            # Match FedBO's x_transform_pre: keep the search bounds fixed and
            # shift the physical input immediately before objective evaluation.
            return y_shift + base_function(x + x_shift)

        objectives.append(
            Objective(
                name=f"{name}_{client_id}",
                dim=dim,
                lower=np.full(dim, lower),
                upper=np.full(dim, upper),
                optimum=base_optimum + y_shift,
                function=shifted,
            )
        )
    return objectives


def _combination_objectives(
    dim: int,
    num_clients: int,
    rng: np.random.Generator,
) -> list[Objective]:
    lower, upper = -5.0, 10.0
    component_functions = (_ackley, _cosine, _rastrigin, _griewank, _sphere)
    shifts = np.array([0.0, -1.0, 8.0, -4.0, 3.0])
    y_mins = np.array([0.0, -dim, 0.0, 0.0, 0.0])
    y_maxs = np.array([22.35, dim, 46.21 * dim, 91.0 * dim, 5.12**2 * dim])
    bounds = [(lower, upper)] * dim
    objectives = []

    for client_id in range(num_clients):
        weights = rng.uniform(size=len(component_functions))
        weights /= weights.sum()

        def combination(x: FloatArray, weights: FloatArray = weights) -> float:
            values = np.array(
                [
                    function(x - shift)
                    for function, shift in zip(component_functions, shifts)
                ]
            )
            scaled = (values - y_mins) / (y_maxs - y_mins)
            return float(weights @ scaled)

        # Combination.best_y in the source benchmark is estimated with 100*d
        # independently initialized L-BFGS-B runs.
        optimum = np.inf
        for _ in range(100 * dim):
            result = minimize(
                combination,
                rng.uniform(lower, upper, size=dim),
                method="L-BFGS-B",
                bounds=bounds,
            )
            optimum = min(optimum, float(result.fun))

        objectives.append(
            Objective(
                name=f"combination_{client_id}",
                dim=dim,
                lower=np.full(dim, lower),
                upper=np.full(dim, upper),
                optimum=optimum,
                function=combination,
            )
        )
    return objectives


def _cartpole_objectives(
    dim: int,
    num_clients: int,
    rng: np.random.Generator,
) -> list[Objective]:
    if dim != 5:
        raise ValueError(f"cartpole requires dim=5, received dim={dim}.")
    objectives = []
    for client_id in range(num_clients):
        gravity = float(rng.uniform(5.0, 20.0))
        mass_cart = float(rng.uniform(0.5, 2.0))
        mass_pole = float(rng.uniform(0.05, 0.5))
        pole_length = float(rng.uniform(0.25, 1.0))
        force_mag = float(rng.uniform(5.0, 20.0))
        total_mass = mass_cart + mass_pole
        denominator = (mass_pole * pole_length) ** 2 - total_mass * (
            mass_pole * pole_length**2 * 4.0 / 3.0
        )
        system = np.zeros((4, 4))
        system[0, 1] = 1.0
        system[2, 3] = 1.0
        system[1, 2] = -(mass_pole**2 * pole_length**2 * gravity) / (
            pole_length * denominator
        )
        system[3, 2] = total_mass * mass_pole * gravity * pole_length / denominator
        control = np.zeros((4, 1))
        control[1, 0] = -(4.0 / 3.0) * mass_pole * pole_length**2 / denominator
        control[3, 0] = mass_pole * pole_length / denominator

        def cartpole(
            x: FloatArray,
            system: FloatArray = system,
            control: FloatArray = control,
            gravity: float = gravity,
            mass_cart: float = mass_cart,
            mass_pole: float = mass_pole,
            pole_length: float = pole_length,
            force_mag: float = force_mag,
        ) -> float:
            try:
                import gymnasium as gym
            except ImportError as exc:
                raise ImportError(
                    "The cartpole task requires the optional Gymnasium classic-control dependency."
                ) from exc

            q = np.diag(x[:4])
            r = np.array([[x[4]]])
            try:
                riccati = solve_continuous_are(system, control, q, r)
                gain = np.linalg.inv(r) @ control.T @ riccati
            except np.linalg.LinAlgError:
                return 0.0

            environment = gym.make("CartPole-v1")
            environment.unwrapped.gravity = gravity
            environment.unwrapped.masscart = mass_cart
            environment.unwrapped.masspole = mass_pole
            environment.unwrapped.length = pole_length
            environment.unwrapped.force_mag = force_mag
            total_steps = 0
            try:
                for _ in range(10):
                    state, _ = environment.reset(seed=int(rng.integers(1_000_000)))
                    for steps in range(500):
                        action = int((-(gain @ state)[0]) > 0.0)
                        state, _, terminated, truncated, _ = environment.step(action)
                        if terminated or truncated:
                            total_steps += steps + 1
                            break
                    else:
                        total_steps += 500
            finally:
                environment.close()
            return -total_steps / 10.0

        objectives.append(
            Objective(
                name=f"cartpole_{client_id}",
                dim=dim,
                lower=np.full(dim, 1e-5),
                upper=np.full(dim, 10.0),
                optimum=-500.0,
                function=cartpole,
            )
        )
    return objectives


def homogeneous_objectives(
    name: str,
    dim: int,
    num_clients: int,
    rng: np.random.Generator,
    heterogeneous_shifts: bool = True,
) -> list[Objective]:
    if name not in _FUNCTIONS:
        raise ValueError(f"Unknown objective {name!r}. Choose one of {sorted(_FUNCTIONS)} or 'manytask'.")
    function, lower, upper = _FUNCTIONS[name]
    objectives = []
    for client_id in range(num_clients):
        shift = rng.uniform(-0.1, 0.1, size=dim) * (upper - lower) if heterogeneous_shifts else np.zeros(dim)

        def shifted(x: FloatArray, shift: FloatArray = shift, function: ObjectiveFunction = function) -> float:
            return function(x - shift)

        objectives.append(
            Objective(
                name=f"{name}_{client_id}",
                dim=dim,
                lower=np.full(dim, lower),
                upper=np.full(dim, upper),
                optimum=0.0,
                function=shifted,
            )
        )
    return objectives


def _manytask_data_dir() -> Path:
    return Path(__file__).resolve().parent / "fmtbo" / "fedmany" / "Tasks" / "50d"


def manytask_objectives(dim: int, num_clients: int) -> list[Objective]:
    if not 1 <= dim <= 50:
        raise ValueError("The ManyTask benchmark supports dimensions 1 through 50.")
    specs = [
        ("griewank", -100, 100, "CI_H.mat", "Rotation_Task1", "GO_Task1"),
        ("rastrigin", -50, 50, "CI_H.mat", "Rotation_Task2", "GO_Task2"),
        ("ackley", -50, 50, "CI_M.mat", "Rotation_Task1", "GO_Task1"),
        ("rastrigin", -50, 50, "CI_M.mat", "Rotation_Task2", "GO_Task2"),
        ("ackley", -50, 50, "CI_L.mat", "Rotation_Task1", "GO_Task1"),
        ("schwefel", -500, 500, None, None, None),
        ("rastrigin", -50, 50, "PI_H.mat", "Rotation_Task1", "GO_Task1"),
        ("sphere", -100, 100, "PI_H.mat", None, "GO_Task2"),
        ("ackley", -50, 50, "PI_M.mat", "Rotation_Task1", "GO_Task1"),
        ("rosenbrock", -50, 50, None, None, None),
        ("ackley", -50, 50, "PI_L.mat", "Rotation_Task1", "GO_Task1"),
        ("weierstrass", -0.5, 0.5, "PI_L.mat", "Rotation_Task2", "GO_Task2"),
        ("rosenbrock", -50, 50, None, None, None),
        ("rastrigin", -50, 50, "NI_H.mat", "Rotation_Task2", "GO_Task2"),
        ("griewank", -100, 100, "NI_M.mat", "Rotation_Task1", "GO_Task1"),
        ("weierstrass", -0.5, 0.5, "NI_M.mat", "Rotation_Task2", "GO_Task2"),
        ("rastrigin", -50, 50, "NI_L.mat", "Rotation_Task1", "GO_Task1"),
        ("schwefel", -500, 500, None, None, None),
    ]
    if not 1 <= num_clients <= len(specs):
        raise ValueError(f"ManyTask supports between 1 and {len(specs)} clients.")
    function_map = {**{key: value[0] for key, value in _FUNCTIONS.items()}, "weierstrass": _weierstrass}
    objectives = []
    for client_id, (name, lower, upper, filename, rotation_key, shift_key) in enumerate(specs[:num_clients]):
        rotation = np.eye(dim)
        shift = np.zeros(dim)
        if filename is not None:
            data = loadmat(_manytask_data_dir() / filename)
            if rotation_key is not None:
                rotation = np.asarray(data[rotation_key][:dim, :dim], dtype=np.float64)
            if shift_key is not None:
                shift = np.asarray(data[shift_key], dtype=np.float64).squeeze()[:dim]
        base_function = function_map[name]

        def transformed(
            x: FloatArray,
            rotation: FloatArray = rotation,
            shift: FloatArray = shift,
            base_function: ObjectiveFunction = base_function,
        ) -> float:
            return base_function(rotation @ (x - shift))

        objectives.append(
            Objective(
                name=f"manytask_{client_id}_{name}",
                dim=dim,
                lower=np.full(dim, lower, dtype=np.float64),
                upper=np.full(dim, upper, dtype=np.float64),
                optimum=0.0,
                function=transformed,
            )
        )
    return objectives


def create_objectives(
    name: str,
    dim: int,
    num_clients: int,
    rng: np.random.Generator,
    heterogeneous_shifts: bool = True,
) -> list[Objective]:
    if name == "manytask":
        return manytask_objectives(dim, num_clients)
    if name in {"ackley_shifted", "branin_shifted", "hartmann6d_shifted"}:
        return _shifted_objectives(name, dim, num_clients, rng)
    if name == "combination":
        return _combination_objectives(dim, num_clients, rng)
    if name == "cartpole":
        return _cartpole_objectives(dim, num_clients, rng)
    return homogeneous_objectives(name, dim, num_clients, rng, heterogeneous_shifts)
