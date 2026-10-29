from pyDOE import lhs
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from scipy import optimize
import gymnasium as gym
import scipy.linalg


def calibrate_x(x_r, x_lb, x_up):
    return np.clip(x_r, x_lb, x_up)


def _checker(method):
    def inner(ref, x, *args, **kwargs):
        if not type(x).__module__ == np.__name__:
            x = np.array(x)
            if len(x.shape) > 1:
                x = np.squeeze(x)
        # dimension check
        assert len(x) == ref.d
        try:
            if ref.normalized:
                # retrieve to the original value
                x = x * (ref.x_ub - ref.x_lb) + ref.x_lb
            if ref.transform is not None:
                # for multi-task problems
                x = ref.transform(x)
            # ensure all the decision variables are located within the predefined boundary
            x = calibrate_x(x, x_lb=ref.x_lb, x_up=ref.x_ub)
            return method(ref, x, *args, **kwargs)
        except Exception as inst:
            if inst.__str__() == "Error":
                print("input out of domain: " + str((ref.x_lb, ref.x_ub)))
                print("input must be vector: [1,2,...]")
            else:
                print(inst)
    return inner


class Transform:
    pass


class MultiTaskTransform(Transform):

    def __init__(self, rot_mat=1, shift_mat=0):
        self.rot_mat = rot_mat
        self.shift_mat = shift_mat

    def __call__(self, x: np.ndarray):
        assert len(x.shape) == 1
        return self._rotate(self._shift(x))

    def _shift(self, x):
        return x - self.shift_mat

    def _rotate(self, x):
        return np.dot(self.rot_mat, x)


class ObjectiveFunction:

    def __init__(self, x_lb, x_ub, d, normalized, transform):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.d = d
        try:
            self.b_dim = len(self.x_lb)
        except:
            self.b_dim = 1
        if self.b_dim > 1:
            assert self.b_dim == len(self.x_ub) == self.d
        self.normalized = normalized
        self.transform = transform

    @property
    def x_bound(self):
        if self.normalized:
            x_bound = [0, 1]
        else:
            if self.b_dim == 1:
                x_bound = [self.x_lb, self.x_ub]
            else:
                x_bound = np.hstack((np.expand_dims(self.x_lb, 1),
                                    np.expand_dims(self.x_ub, 1)))
        return x_bound

    def __repr__(self):
        return type(self).__name__

    def init_x_with_bound(self, size, iid, pb=None):
        """
        Initialize random decision variables within the (partitioned) boundary
        :param size: number of init samples
        :param iid: if using iid split method
        :param pb: partition boundary
        :return: initialized decision variables with shape size x d
        """
        if iid:
            # LHS sampling
            result = lhs(self.d, samples=size)
        else:
            result = np.random.uniform(pb[0], pb[1], size=(size, self.d))
        if self.normalized:
            return result
        else:
            # element wise multiplication
            return result * (self.x_ub - self.x_lb) + self.x_lb

    def random_uniform_x(self, size, iid, pb=None):
        if iid:
            out = np.random.uniform(low=0, high=1, size=(size, self.d))
        else:
            out = np.random.uniform(pb[0], pb[1], size=(size, self.d))
        if not self.normalized:
            out = out * (self.x_ub - self.x_lb) + self.x_lb
        return out

    @_checker
    def __call__(self, x, *args, **kwargs):
        raise NotImplementedError


class Griewank(ObjectiveFunction):

    def __init__(self, x_lb=-600, x_ub=600, d=10, normalized=False, transform=None):
        super(Griewank, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        F1 = 0
        F2 = 1
        d_x = len(x)
        for i in range(0, d_x):
            # z = x[i] - griewank[i]
            z = x[i]
            F1 = F1 + (z ** 2 / 4000)
            F2 = F2 * (np.cos(z / np.sqrt(i + 1)))
        return F1 - F2 + 1


class Rastrigin(ObjectiveFunction):

    def __init__(self, x_lb=-5, x_ub=5, d=10, normalized=False, transform=None):
        super(Rastrigin, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        F = 0
        d_x = len(x)
        for i in range(0, d_x):
            z = x[i]
            F = F + (z ** 2 - 10 * np.cos(2 * np.pi * z) + 10)
        return F


class Ackley(ObjectiveFunction):

    def __init__(self, x_lb=-32.768, x_ub=32.768, d=10, normalized=False, transform=None):
        super(Ackley, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        sum1 = 0
        sum2 = 0
        d_x = len(x)
        # M =1 no rotated
        M = 1
        for i in range(0, d_x):
            # z = M * (x[i] - ackley[i])
            z = x[i]
            sum1 = sum1 + z ** 2
            sum2 = sum2 + np.cos(2 * np.pi * z)
        out = -20 * np.exp(-0.2 * np.sqrt(sum1 / d_x)) - np.exp(sum2 / d_x) + 20 + np.e
        # F += f_bias[5]
        return out


class Schwefel(ObjectiveFunction):

    def __init__(self, x_lb=-500, x_ub=500, d=10, normalized=False, transform=None):
        super(Schwefel, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        out = 0
        d_x = len(x)
        for i in range(d_x):
            out += x[i] * np.sin(np.sqrt(abs(x[i])))
        return 418.9829 * d_x - out


class Sphere(ObjectiveFunction):

    def __init__(self, x_lb=-100, x_ub=100, d=10, normalized=False, transform=None):
        super(Sphere, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        return x.dot(x)


class Rosenbrock(ObjectiveFunction):
    def __init__(self, x_lb=-2.048, x_ub=2.048, d=10, normalized=False, transform=None):
        super(Rosenbrock, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        out = 0
        d_x = len(x)
        for i in range(0, d_x - 1):
            tmp = 100 * np.power(x[i + 1] ** 2 - x[i], 2) + np.power(x[i] - 1, 2)
            out += tmp
        # F += f_bias[2]
        return out


class Weierstrass(ObjectiveFunction):

    def __init__(self, x_lb=-0.5, x_ub=0.5, d=10, normalized=False, transform=None):
        super(Weierstrass, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        D = len(x)
        a = 0.5
        b = 3
        kmax = 21
        obj = 0
        for i in range(D):
            for k in range(kmax):
                obj += a**k * np.cos(2 * np.pi * b**k * (x[i] + 0.5))
        for k in range(kmax):
            obj -= D * a**k * np.cos(2 * np.pi * b**k * 0.5)
        return obj


class Ellipsoid(ObjectiveFunction):

    def __init__(self, x_lb=-5.12, x_ub=5.12, d=10, normalized=False, transform=None):
        super(Ellipsoid, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        out = 0.0
        d = len(x)
        d_L = [i for i in range(1, d + 1)]
        for i_ in d_L:
            out += i_ * x[i_ - 1] ** 2
        return out


class SvmLandmine(ObjectiveFunction):

    def __init__(self, x_lb=np.array([1e-4, 1e-2]), x_ub=np.array([10.0, 10.0]), d=None, normalized=True, transform=None):
        assert len(x_lb.shape) == len(x_ub.shape) == 1
        super(SvmLandmine, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, x_train, y_train, x_test, y_test):
        # training svm and get the prediction with real datasets
        clf = svm.SVC(kernel="rbf", C=x[0], gamma=x[1], probability=True)
        clf.fit(x_train, y_train)
        pred = clf.predict_proba(x_test)
        score = roc_auc_score(y_test, pred[:, 1])
        return score


class Cosine(ObjectiveFunction):

    def __init__(self, x_lb=-np.pi * 2, x_ub=np.pi * 2, d=10, normalized=False, transform=None):
        super(Cosine, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        return np.sum(np.cos(x) * ((0.1 / (2 * np.pi)) * np.abs(x) - 1))


class Levy(ObjectiveFunction):

    def __init__(self, x_lb=-10.0, x_ub=10, d=10, normalized=False, transform=None):
        super(Levy, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        w = 1.0 + (x - 1.0) / 4.0
        # First term
        term1 = np.sin(np.pi * w[0]) ** 2
        # Middle summation term
        w_mid = w[:-1]
        term2 = np.sum(
            (w_mid - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w_mid + 1.0) ** 2)
        )
        # Last term
        w_last = w[-1]
        term3 = (w_last - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w_last) ** 2)
        result = term1 + term2 + term3
        return result


class Zakharov(ObjectiveFunction):

    def __init__(self, x_lb=-5, x_ub=10, d=10, normalized=False, transform=None):
        super(Zakharov, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x: np.ndarray, *args, **kwargs):
        input_dim = x.shape[0]
        # First term: sum(x_i^2)
        term1 = np.sum(x**2)
        # Second term calculation: sum(0.5 * i * x_i)
        indices = np.arange(1, input_dim + 1)
        term2_sum = np.sum(0.5 * indices * x)
        # Combine terms
        result = term1 + term2_sum**2 + term2_sum**4
        return result


def minmax_scaling(X: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """Applies Min-Max scaling to transform X to [0, 1] range."""
    return (X - lb) / (ub - lb)


def minimize_func(
    func,
    input_dim: int,
    bounds: tuple[float, float],
    num_restarts: int = 10,
):
    lb, ub = bounds

    X_Cand = np.zeros((num_restarts, input_dim))
    Y_Cand = np.zeros((num_restarts, 1))

    def f(x):
        return float(func(x).squeeze())

    def grad_func(x):
        eps = np.sqrt(np.finfo(float).eps)
        return optimize.approx_fprime(x, f, epsilon=eps)

    for j in range(num_restarts):
        x0 = lb + (ub - lb) * np.random.uniform(size=(input_dim,))

        res = optimize.minimize(
            f, x0, jac=grad_func, bounds=[(lb, ub)] * input_dim, method="L-BFGS-B"
        )

        X_Cand[j, :] = res.x
        Y_Cand[j] = res.fun

    X_Cand = np.clip(X_Cand, lb, ub)
    best_idx = np.argmin(Y_Cand)
    best_x = X_Cand[best_idx]
    best_y = Y_Cand[best_idx]
    return best_x, best_y


class Combination(ObjectiveFunction):

    def __init__(self, x_lb=-5, x_ub=10, d=10, normalized=False):
        super().__init__(x_lb, x_ub, d, normalized, None)
        self.objectives = [
            Ackley(self.x_lb, self.x_ub, self.d, self.normalized),
            Cosine(self.x_lb, self.x_ub, self.d, self.normalized),
            Rastrigin(self.x_lb, self.x_ub, self.d, self.normalized),
            Griewank(self.x_lb, self.x_ub, self.d, self.normalized),
            Sphere(self.x_lb, self.x_ub, self.d, self.normalized),
        ]
        num_objectives = len(self.objectives)

        weights = np.random.uniform(low=0.0, high=1.0, size=(num_objectives,))
        self.weights = weights / weights.sum()

        self.shifts = [0, -1, 8, -4, 3]

        self.y_mins = np.array([0, -self.d, 0, 0, 0])
        self.y_maxs = np.array([22.35, self.d, 46.21 * self.d,
                               91 * self.d, 5.12**2 * d])
        self.best_x, self.best_y = minimize_func(
            self, self.d, (self.x_lb, self.x_ub), 100 * self.d)

    @_checker
    def __call__(self, x, *args, **kwargs):
        return sum(
            w * minmax_scaling(obj(x - shift), y_min, y_max)
            for w, obj, shift, y_min, y_max in zip(self.weights, self.objectives, self.shifts, self.y_mins, self.y_maxs)
        )


class AckleyShifted(Ackley):

    def __init__(self, x_lb=-5, x_ub=10, d=10, normalized=False, transform=None):
        super().__init__(x_lb, x_ub, d, normalized, transform)
        self.x_shift = np.random.uniform(low=-3, high=3)
        self.y_shift = np.random.uniform(low=-3, high=3)
        self.best_y = 0.0 + self.y_shift
        # Shift the search box with the objective so x + x_shift always lies
        # in Ackley's original [x_lb, x_ub] domain.
        self.x_lb = x_lb - self.x_shift
        self.x_ub = x_ub - self.x_shift

    @_checker
    def __call__(self, x, *args, **kwargs):
        shifted_x = x + self.x_shift
        sum1 = np.sum(shifted_x ** 2)
        sum2 = np.sum(np.cos(2 * np.pi * shifted_x))
        return (
            self.y_shift
            - 20 * np.exp(-0.2 * np.sqrt(sum1 / self.d))
            - np.exp(sum2 / self.d)
            + 20
            + np.e
        )


class Branin(ObjectiveFunction):

    def __init__(self, x_lb=0, x_ub=10, normalized=False, transform=None):
        super().__init__(x_lb, x_ub, 2, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        return self._evaluate(x)

    @staticmethod
    def _evaluate(x):
        X1 = x[0]
        X2 = x[1]
        result = (
            (X2 - (5.1 / (4 * np.pi**2)) * X1**2 + (5 / np.pi) * X1 - 6) ** 2
            + 10 * (1 - (1 / (8 * np.pi))) * np.cos(X1)
            + 10
        )
        return result


class BraninShifted(Branin):

    def __init__(self, x_lb=0, x_ub=10, normalized=False, transform=None):
        super().__init__(x_lb, x_ub, normalized, transform)
        self.x_shift = np.random.uniform(low=-3, high=3)
        self.y_shift = np.random.uniform(low=-3, high=3)
        self.best_y = 0.397887 + self.y_shift
        # Shift the search box with the objective so x + x_shift always lies
        # in Branin's original [x_lb, x_ub] domain.
        self.x_lb = x_lb - self.x_shift
        self.x_ub = x_ub - self.x_shift

    @_checker
    def __call__(self, x, *args, **kwargs):
        return self.y_shift + self._evaluate(x + self.x_shift)


class Hartmann6d(ObjectiveFunction):

    def __init__(self, x_lb=0, x_ub=1, normalized=False, transform=None):
        super().__init__(x_lb, x_ub, 6, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        return self._evaluate(x)

    @staticmethod
    def _evaluate(x):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ]
        )
        P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

        x = x[None, :]
        inner_sum = -np.sum(A * (x - P) ** 2, axis=1)
        result = -np.sum(alpha * np.exp(inner_sum), axis=0)
        return result


class Hartmann6dShifted(Hartmann6d):

    def __init__(self, x_lb=0, x_ub=1, normalized=False, transform=None):
        super().__init__(x_lb, x_ub, normalized, transform)
        self.x_shift = np.random.uniform(low=-0.3, high=0.15)
        self.y_shift = np.random.uniform(low=-3, high=3)
        self.best_y = -3.32237 + self.y_shift
        # Shift the search box with the objective so x + x_shift always lies
        # in Hartmann-6's original [x_lb, x_ub] domain.
        self.x_lb = x_lb - self.x_shift
        self.x_ub = x_ub - self.x_shift

    @_checker
    def __call__(self, x, *args, **kwargs):
        return self.y_shift + self._evaluate(x + self.x_shift)


class CartpoleLQRObjective(ObjectiveFunction):
    """
    An objective function for optimizing the LQR controller of a CartPole system,
    using the standard Gymnasium environment for simulation.
    """

    def __init__(self):
        super().__init__(1e-5, 10.0, 5, False, None)

        self.gravity = self.generate_random_float(5.0, 20.0)
        self.mass_cart = self.generate_random_float(0.5, 2.0)
        self.mass_pole = self.generate_random_float(0.05, 0.5)
        self.total_mass = self.mass_cart + self.mass_pole
        # Half the pole's length
        self.pole_length = self.generate_random_float(0.25, 1.0)
        self.pole_mass_length = self.mass_pole * self.pole_length
        self.denominator = self.pole_mass_length**2 - self.total_mass * (
            self.mass_pole * self.pole_length**2 * 4 / 3
        )
        self.force_mag = self.generate_random_float(5.0, 20.0)

        self.A = np.zeros((4, 4))
        self.A[0, 1] = 1.0
        self.A[2, 3] = 1.0
        self.A[1, 2] = (
            -(self.mass_pole**2) * self.pole_length**2 * self.gravity
        ) / (self.pole_length * self.denominator)
        self.A[3, 2] = (
            self.total_mass * self.mass_pole * self.gravity * self.pole_length
        ) / self.denominator

        self.B = np.zeros((4, 1))
        self.B[1, 0] = (
            -(4 / 3) * self.mass_pole * self.pole_length**2
        ) / self.denominator
        self.B[3, 0] = (self.pole_mass_length) / self.denominator

        self.max_steps = 500
        self.num_eval_episodes = 10

    @_checker
    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
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
        q_diag = x[:4]
        r_val = x[4]
        # The cost is negated, so higher performance (survival time) is better.
        return -self.evaluate_lqr_performance(q_diag, r_val)

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
        seed = np.random.randint(size=(), low=0, high=1e6)
        return int(seed)

    def generate_random_float(self, lb: float, ub: float):
        return float(np.random.uniform(size=(), low=lb, high=ub))

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
