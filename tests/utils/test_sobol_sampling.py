from jax import random
import jax.numpy as jnp
import numpy as np
from scipy.stats.qmc import Sobol
import matplotlib.pyplot as plt


if __name__ == "__main__":
    key = random.key(0)
    key, subkey = random.split(key)
    np_rng1 = np.random.default_rng(np.asarray(random.key_data(subkey)))
    np_rng2 = np.random.default_rng(np.asarray(random.key_data(subkey)))

    input_dim = 1
    num_samples = 8

    # Create and scale Sobol samples
    sobol_engine = Sobol(d=input_dim, scramble=True, rng=np_rng1)
    sobol_samples = sobol_engine.random(n=num_samples)
    print(sobol_samples)

    # # Scale to bounds
    lower, upper = (-5, 10)
    X1 = jnp.array(lower + sobol_samples * (upper - lower))
    Y1 = jnp.ones_like(X1)

    sobol_engine = Sobol(d=input_dim, scramble=True, rng=np_rng2)
    m = int(np.log2(num_samples))
    print("m ", m)
    sobol_samples = sobol_engine.random_base2(m=m)
    print(sobol_samples)
    lower, upper = (-5, 10)
    X2 = jnp.array(lower + sobol_samples * (upper - lower))
    Y2 = jnp.ones_like(X2)

    plt.scatter(X1, Y1)
    plt.scatter(X2, Y2)
    plt.show()
