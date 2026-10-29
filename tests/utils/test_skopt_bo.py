import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import time
import jax
import jax.numpy as jnp

from fedbo.data.objectives import AckleyObjective
from fedbo.transformations import MinMaxScaling, RandomShifting


if __name__ == "__main__":
    for i in range(30):
        random_state = i
        np.random.seed(random_state)
        key = jax.random.key(random_state)

        dim = 10
        objective = AckleyObjective(dim)
        minmax = MinMaxScaling(objective.bounds())
        key, subkey = jax.random.split(key)
        shifting = RandomShifting((-3, 3), subkey)

        search_space = [Real(*objective.bounds(), name=f"x{i}") for i in range(dim)]

        # --- Wrapper for the objective function for skopt ---
        @use_named_args(search_space)
        def objective_function(**params):
            """
            This wrapper function is called by skopt. It unpacks the named parameters
            and calls our evaluation function. skopt minimizes, so we return the
            negative of our score (average survival time).
            """
            x = np.array(list(params.values()))
            # x = shifting(x)
            score = float(objective(x).squeeze())

            # print(f"Params: {params} -> Score: {score}")

            return score

        print("--- Starting Bayesian Optimization ---")
        start_time = time.time()

        # Run the optimization
        # n_calls: Total number of evaluations.
        # n_initial_points: How many random points to sample before using the model.
        result = gp_minimize(
            func=objective_function,
            dimensions=search_space,
            n_calls=50,
            n_initial_points=dim,
            random_state=random_state,
            acq_func="EI",  # Expected Improvement acquisition function
        )

        end_time = time.time()
        print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")

        # --- Print and use the best results ---
        best_params = result.x
        best_score = result.fun

        print("\n--- Best Parameters Found ---")
        print(f"Params: {best_params}")
        print(f"Score: {best_score}")
