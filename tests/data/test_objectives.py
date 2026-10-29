import jax
import jax.numpy as jnp

from fedbo.data.objectives import (
    ackley_objective,
    branin_objective,
    cosine_objective,
    hartmann6d_objective,
    rastrigin_objective,
    griewank_objective,
    sphere_objective,
)

if __name__ == "__main__":

    objectives = [
        ackley_objective,
        branin_objective,
        cosine_objective,
        rastrigin_objective,
        hartmann6d_objective,
        griewank_objective,
        sphere_objective,
    ]
    min_inputs = [
        jnp.zeros(shape=(1, 10)),
        jnp.array([[-jnp.pi, 12.275], [jnp.pi, 2.275], [9.42478, 2.475]]),
        jnp.zeros(shape=(1, 8)),
        jnp.zeros(shape=(1, 4)),
        jnp.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]),
        jnp.zeros(shape=(1, 9)),
        jnp.zeros(shape=(1, 6)),
    ]
    min_values = [
        jnp.array([0]),
        jnp.array([[0.397887] * 3]),
        jnp.array([-8]),
        jnp.array([0]),
        jnp.array([-3.32237]),
        jnp.array([0]),
        jnp.array([0]),
    ]

    for obj, min_in, min_val in zip(objectives, min_inputs, min_values):
        obj_vmap = jax.vmap(obj)
        obj_val = obj_vmap(min_in)
        print("Objective value: ", obj_val)
        print("Min value: ", min_val)
        assert jnp.allclose(obj_val, min_val, rtol=1e-5, atol=1e-8)
