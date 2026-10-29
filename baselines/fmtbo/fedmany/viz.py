import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from objectives import create_objectives


objective = create_objectives(
    "ackley", 1, 1, np.random.default_rng(0)
)[0]


# Create grid
x = np.linspace(objective.x_lb, objective.x_ub, 1000)[:, None]
y = np.zeros_like(x)
for i in range(x.shape[0]):
    y[i] = objective(x[i])
plt.title("Objective Function")
# plt.scatter(min_x, min_y, color='red', marker="x")
plt.plot(x, y, label="Y")
plt.legend()
plt.show()
