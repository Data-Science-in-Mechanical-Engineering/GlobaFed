import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern

import importlib.resources

style_package = "fedbo.visualization.style"
style_file = "fedbo.mplstyle"
try:
    file_ref = importlib.resources.files(style_package).joinpath(style_file)

    with importlib.resources.as_file(file_ref) as path:
        plt.style.use(path)

except (ModuleNotFoundError, FileNotFoundError):
    print(f"Warning: Could not load the style '{style_file}' from '{style_package}'.")

np.random.seed(1)


def f_ackley(x, a=20, b=0.2, c=2 * np.pi):
    """
    The 1D Ackley function.
    Input x is expected to be shape (n, 1).
    """
    term1 = -a * np.exp(-b * np.abs(x))
    term2 = -np.exp(np.cos(c * x))

    return term1 + term2 + a + np.exp(1)


X_domain = np.linspace(-10, 10, 300).reshape(-1, 1)
y_true = f_ackley(X_domain)

X_train = np.array([-4.5, -3.0, 4.0]).reshape(-1, 1)
y_train = f_ackley(X_train)

kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10, normalize_y=True)

n_iterations = 6

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i in range(n_iterations):

    gp.fit(X_train, y_train)
    y_mean, y_std = gp.predict(X_domain, return_std=True)

    f_sample = gp.sample_y(X_domain, 1).ravel()
    next_x_index = np.argmin(f_sample)
    next_x = X_domain[next_x_index].reshape(1, -1)

    next_y = f_ackley(next_x)

    if i >= 3:
        plot_index = i - 3

        ax = axes[plot_index]

        ax.plot(X_domain, y_true, "k-", label="Objective")
        ax.plot(X_domain, y_mean, label="Surrogate Mean")

        ax.fill_between(
            X_domain.ravel(), (y_mean - 1.96 * y_std), (y_mean + 1.96 * y_std), alpha=0.2, label="Surrogate Uncertainty"
        )

        ax.plot(X_domain, f_sample, "g:", label="Thompson Sample")

        ax.plot(X_train, y_train, "ko", markersize=5)

        ax.plot(next_x, next_y, "rx", markersize=8, mew=3, label="Next Sample")

        ax.set_ylim(-3, 23)
        ax.set_xlim(-10.5, 10.5)
        ax.set_xlabel("x")

        if plot_index == 0:
            ax.set_ylabel("f(x)")

    X_train = np.vstack((X_train, next_x))
    y_train = np.vstack((y_train, next_y))


handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=5)

plt.tight_layout()
plt.subplots_adjust(top=0.85)  

fig.savefig("bo_gp_thompson.pdf")
plt.show()

best_idx = np.argmin(y_train)
print(f"\nOptimization Finished (after {n_iterations} steps).")
print(f"Best point found: x = {X_train[best_idx][0]:.4f}, y = {y_train[best_idx][0]:.4f}")
