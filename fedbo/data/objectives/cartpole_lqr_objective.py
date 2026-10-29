import jax.numpy as jnp
import numpy as np
import scipy.linalg
import gymnasium as gym
from jax import random
from jaxtyping import Array, Key

from fedbo.data.objectives.base_objective import BaseObjective


class CartpoleLQRObjective(BaseObjective):
    """
    An objective function for optimizing the LQR controller of a CartPole system,
    using the standard Gymnasium environment for simulation.
    """

    def __init__(self, key: Key):
        # Create the Gymnasium environment
        self.key = key

        self.gravity = self.generate_random_float(5.0, 20.0)
        self.mass_cart = self.generate_random_float(0.5, 2.0)
        self.mass_pole = self.generate_random_float(0.05, 0.5)
        self.total_mass = self.mass_cart + self.mass_pole
        # Half the pole's length
        self.pole_length = self.generate_random_float(0.25, 1.0)
        self.pole_mass_length = self.mass_pole * self.pole_length
        self.denominator = self.pole_mass_length**2 - self.total_mass * (self.mass_pole * self.pole_length**2 * 4 / 3)
        self.force_mag = self.generate_random_float(5.0, 20.0)

        print(f"Gravity: {self.gravity}")
        print(f"Mass Cart: {self.mass_cart}")
        print(f"Mass pole: {self.mass_pole}")
        print(f"Pole Length: {self.pole_length}")
        print(f"Force Mag: {self.force_mag}")

        self.A = np.zeros((4, 4))
        self.A[0, 1] = 1.0
        self.A[2, 3] = 1.0
        self.A[1, 2] = (-(self.mass_pole**2) * self.pole_length**2 * self.gravity) / (self.pole_length * self.denominator)
        self.A[3, 2] = (self.total_mass * self.mass_pole * self.gravity * self.pole_length) / self.denominator

        self.B = np.zeros((4, 1))
        self.B[1, 0] = (-(4 / 3) * self.mass_pole * self.pole_length**2) / self.denominator
        self.B[3, 0] = (self.pole_mass_length) / self.denominator

        self.max_steps = 500
        self.num_eval_episodes = 10
        super().__init__(input_dim=5, min_y=-float(self.max_steps), max_y=0.0)

    def __call__(self, X: Array) -> Array:
        """
        Evaluates a batch of LQR parameters and returns their costs.

        Args:
            X: A JAX array of shape (batch_size, 5) where each row
                        is a set of [q1, q2, q3, q4, r1].

        Returns:
            A JAX array of shape (batch_size,) containing the cost for each
            parameter set.
        """
        # Use a list comprehension to iterate over the batch
        costs = [self._evaluate_single(x) for x in X]
        return jnp.array(costs)

    def _evaluate_single(self, x: Array) -> float:
        """Helper function to evaluate a single set of LQR parameters."""
        q_diag = x[:4]
        r_val = x[4]
        # The cost is negated, so higher performance (survival time) is better.
        return [-self.evaluate_lqr_performance(q_diag, r_val)]

    @staticmethod
    def bounds() -> tuple[float, float]:
        """Defines the search space for the LQR weights."""
        return (1e-5, 10.0)

    def lqr_controller(self, Q, R):
        """Solves the continuous-time LQR controller for a linear system."""
        P = scipy.linalg.solve_continuous_are(self.A, self.B, Q, R)
        K = np.linalg.inv(R) @ self.B.T @ P
        return K

    def make_eval_env(self, render_mode="rgb_array"):
        env = gym.make("CartPole-v1", render_mode=render_mode)
        env.unwrapped.gravity = self.gravity
        env.unwrapped.masscart = self.mass_cart
        env.unwrapped.masspole = self.mass_pole
        env.unwrapped.length = self.pole_length
        env.unwrapped.force_mag = self.force_mag
        return env

    def generate_seed(self):
        self.key, subkey = random.split(self.key)
        seed = random.randint(subkey, shape=(), minval=0, maxval=1e6)
        return int(seed)

    def generate_random_float(self, lb: float, ub: float):
        self.key, subkey = random.split(self.key)
        return float(random.uniform(subkey, shape=(), minval=lb, maxval=ub))

    def evaluate_lqr_performance(self, Q_diag, R_val):
        """
        Runs a simulation with given LQR parameters and returns the average survival time.
        """
        # Construct Q and R matrices
        Q = np.diag(Q_diag)
        R = np.array([[R_val]])

        # Calculate LQR gain. Handle potential errors if matrices are invalid.
        try:
            K = self.lqr_controller(Q, R)
        except np.linalg.LinAlgError:
            # Solver failed, meaning parameters are likely unstable. Return worst score.
            return 0.0

        # --- Run multiple episodes to get a robust performance measure ---
        eval_env = self.make_eval_env()
        total_steps = 0

        for _ in range(self.num_eval_episodes):
            state, _ = eval_env.reset(seed=self.generate_seed())
            desired_state = np.array([0.0, 0.0, 0.0, 0.0])

            # Max steps per episode for CartPole-v1 is 500
            for steps in range(500):
                error = state - desired_state
                force = -(K @ error)[0]
                action = 1 if force > 0 else 0

                state, _, terminated, truncated, _ = eval_env.step(action)

                if terminated or truncated:
                    total_steps += steps + 1
                    break
            else:  # This else corresponds to the for loop, runs if loop wasn't 'break'ed
                total_steps += 500

        eval_env.close()
        return total_steps / self.num_eval_episodes


# class CartpoleLQRObjectiveJax(BaseObjective):
#     """
#     An objective function for optimizing the LQR controller of a CartPole system.

#     The function takes a 5-dimensional input vector `X` which corresponds to the
#     diagonal elements of the Q and R matrices for the LQR controller. It simulates
#     the CartPole and returns a cost (negative reward). The goal is to **minimize** this cost.

#     The input `X` consists of:
#     - 4 elements for the diagonal of the state cost matrix Q.
#     - 1 element for the control cost matrix R.
#     """

#     def __init__(self, key: Key):
#         self.key = key
#         # Constants for the simulation
#         self.gravity = 9.8
#         self.mass_cart = 1.0
#         self.mass_pole = 0.1
#         self.total_mass = self.mass_cart + self.mass_pole
#         self.pole_length = 0.5  # actually half the pole's length
#         self.pole_mass_length = self.mass_pole * self.pole_length
#         self.denominator = self.pole_mass_length**2 - self.total_mass * (
#             self.mass_pole * self.pole_length**2 * 4 / 3
#         )

#         self.force_mag = 10.0
#         self.max_steps = 500

#         # Failure thresholds
#         self.theta_threshold_radians = 12 * 2 * jnp.pi / 360
#         self.x_threshold = 2.4

#         # Linearized system matrices (A, B) around the upright position (x=0, theta=0)
#         A = np.zeros((4, 4))
#         A[0, 1] = 1.0
#         A[2, 3] = 1.0
#         A[1, 2] = (-(self.mass_pole**2) * self.pole_length**2 * self.gravity) / (
#             self.pole_length * self.denominator
#         )
#         A[3, 2] = (
#             self.total_mass * self.mass_pole * self.gravity * self.pole_length
#         ) / self.denominator
#         self.A = A

#         B = np.zeros((4, 1))
#         B[1, 0] = (-(4 / 3) * self.mass_pole * self.pole_length**2) / self.denominator
#         B[3, 0] = (self.pole_mass_length) / self.denominator
#         self.B = B

#         self._eval = jax.jit(self._evaluate_single_lqr)

#         # For minimization, the objective is -reward.
#         super().__init__(input_dim=5, min_y=-float(self.max_steps), max_y=0.0)

#     def __call__(self, x: Array, key: Key) -> Array:
#         """
#         Evaluates LQR parameters and returns their cost.
#         """
#         rewards = self._eval(x, key)
#         return jnp.array([-rewards])

#     @staticmethod
#     def bounds() -> tuple[float, float]:
#         """
#         Defines the search space for the LQR weights.
#         """
#         # low = jnp.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-3])
#         # high = jnp.array([10.0, 10.0, 10.0, 10.0, 2.0])
#         return (1e-5, 10.0)

#     def _solve_are_scipy(self, Q, R):
#         """Helper function to wrap the SciPy call."""
#         # SciPy works with NumPy arrays, so we convert JAX arrays.
#         A_np = np.asarray(self.A)
#         B_np = np.asarray(self.B)
#         Q_np = np.asarray(Q)
#         R_np = np.asarray(R)
#         S_np: np.ndarray = scipy.linalg.solve_continuous_are(A_np, B_np, Q_np, R_np)
#         return S_np.astype(np.float32)

#     def _evaluate_single_lqr(self, x_params: Array, key: Key) -> float:
#         """Simulates one episode for a single set of LQR parameters."""
#         # Construct Q and R matrices
#         q_diag = x_params[:4]
#         r_val = x_params[4]
#         Q = jnp.diag(q_diag)
#         R = jnp.array([[r_val]])

#         # Solve the ARE using the SciPy function wrapped in pure_callback
#         # This allows a non-JAX function to be used inside JIT/vmap.
#         S = jax.pure_callback(
#             self._solve_are_scipy,  # The function to call
#             jax.ShapeDtypeStruct(
#                 (4, 4), Q.dtype
#             ),  # The expected output shape and dtype
#             Q,
#             R,  # The arguments to the function
#             vmap_method="sequential",
#         )

#         K = jnp.linalg.inv(R) @ self.B.T @ S

#         # Define the simulation step function
#         def step(state, _):
#             x, _, theta, _ = state
#             done = (jnp.abs(x) > self.x_threshold) | (
#                 jnp.abs(theta) > self.theta_threshold_radians
#             )

#             u = -K @ state
#             force = jnp.clip(u[0], -self.force_mag, self.force_mag)
#             costheta = jnp.cos(theta)
#             sintheta = jnp.sin(theta)
#             temp = (
#                 force + self.pole_mass_length * state[3] ** 2 * sintheta
#             ) / self.total_mass
#             theta_acc = (self.gravity * sintheta - costheta * temp) / (
#                 self.pole_length
#                 * (4.0 / 3.0 - self.mass_pole * costheta**2 / self.total_mass)
#             )
#             x_acc = (
#                 temp - self.pole_mass_length * theta_acc * costheta / self.total_mass
#             )
#             dt = 0.02
#             next_state = state + dt * jnp.array([state[1], x_acc, state[3], theta_acc])

#             reward = jax.lax.select(done, 0.0, 1.0)
#             final_state = jax.lax.select(done, state, next_state)

#             return final_state, reward

#         # Run the simulation
#         initial_state = jax.random.uniform(key, shape=(4,), minval=-0.05, maxval=0.05)
#         _, rewards_per_step = jax.lax.scan(
#             step, initial_state, None, length=self.max_steps
#         )

#         return jnp.sum(rewards_per_step)
