import jax
from jax import random
import jax.numpy as jnp

from fedbo.metrics import InstantaneousRegret, CumulativeRegret, SimpleRegret
from fedbo.data import ObjectiveDataset
from fedbo.data.objectives import AckleyObjective

if __name__ == "__main__":
    key = random.key(0)
    num_samples = 0
    in_dim = 1
    key, dataset_key = random.split(key)
    obj_func = AckleyObjective(in_dim)
    dataset = ObjectiveDataset(obj_func, num_samples, dataset_key)
    for i in range(4):
        dataset.add_data_point(jnp.array((i,)))
    min_value = dataset.min_value
    X = jnp.array((0, 1, 2, 3, 7)).reshape((5, 1))
    f_X = jax.vmap(obj_func)(X)

    latest_x = jnp.array((7,))
    latest_fx = obj_func(latest_x)
    dataset.add_data_point(latest_x)
    i_regret = InstantaneousRegret()
    assert i_regret(dataset.f_X, dataset.min_value) == latest_fx - min_value

    c_regret = CumulativeRegret()
    assert c_regret(dataset.f_X, dataset.min_value) == jnp.sum(f_X - min_value)

    diffs = f_X - min_value
    s_regret = SimpleRegret()
    assert s_regret(dataset.f_X, dataset.min_value) == jnp.min(diffs)
