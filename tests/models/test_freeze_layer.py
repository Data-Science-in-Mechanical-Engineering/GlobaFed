import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import optax


# Toy data
def get_data(dataset_size, *, key):
    x = jrandom.normal(key, (dataset_size, 1))
    y = 5 * x - 2
    return x, y


# Toy data
def get_data(dataset_size, *, key):
    x = jrandom.normal(key, (dataset_size, 1))
    y = 5 * x - 2
    return x, y


# Toy dataloader
def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    dataset_size=10000,
    batch_size=256,
    learning_rate=3e-3,
    steps=1000,
    width_size=8,
    depth=1,
    seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    data = get_data(dataset_size, key=data_key)
    data_iter = dataloader(data, batch_size, key=loader_key)

    # Step 1
    model = eqx.nn.MLP(in_size=1, out_size=1, width_size=width_size, depth=depth, key=model_key)

    # Step 2
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda tree: (tree.layers[-1].weight, tree.layers[-1].bias),
        filter_spec,
        replace=(True, True),
    )

    # Step 3
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        @eqx.filter_grad
        def loss(diff_model, static_model, x, y):
            model = eqx.combine(diff_model, static_model)
            pred_y = jax.vmap(model)(x)
            return jnp.mean((y - pred_y) ** 2)

        diff_model, static_model = eqx.partition(model, filter_spec)
        grads = loss(diff_model, static_model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    original_model = model
    optim = optax.sgd(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), data_iter):
        model, opt_state = make_step(model, x, y, opt_state)
    print(f"Parameters of first layer at initialisation:\n" f"{jtu.tree_leaves(original_model.layers[0])}\n")
    print(f"Parameters of first layer at end of training:\n" f"{jtu.tree_leaves(model.layers[0])}\n")
    print(f"Parameters of last layer at initialisation:\n" f"{jtu.tree_leaves(original_model.layers[-1])}\n")
    print(f"Parameters of last layer at end of training:\n" f"{jtu.tree_leaves(model.layers[-1])}\n")


main()
