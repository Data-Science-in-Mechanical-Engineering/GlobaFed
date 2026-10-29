import gymnasium as gym
import numpy as np
import scipy.linalg
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import time
import jax

from fedbo.data.objectives import CartpoleLQRObjective


if __name__ == "__main__":
    for i in range(40):
        random_state = i
        np.random.seed(random_state)
        key = jax.random.key(random_state)

        objective = CartpoleLQRObjective(key)

        search_space = [
            Real(1e-5, 10, "uniform", name="q_pos"),
            Real(1e-5, 10, "uniform", name="q_vel"),
            Real(1e-5, 10, "uniform", name="q_ang"),
            Real(1e-5, 10, "uniform", name="q_ang_vel"),
            Real(1e-5, 10, "uniform", name="r_cost"),
        ]

        # --- Wrapper for the objective function for skopt ---
        @use_named_args(search_space)
        def objective_function(**params):
            """
            This wrapper function is called by skopt. It unpacks the named parameters
            and calls our evaluation function. skopt minimizes, so we return the
            negative of our score (average survival time).
            """
            Q_diag = [
                params["q_pos"],
                params["q_vel"],
                params["q_ang"],
                params["q_ang_vel"],
            ]
            R_val = params["r_cost"]

            score = float(objective(np.array([list(params.values())])).squeeze())

            print(f"Params: Q_diag={Q_diag}, R={R_val:.4f} -> Avg Score: {score:.2f}")

            return score

        print("--- Starting Bayesian Optimization for LQR Tuning ---")
        start_time = time.time()

        # Run the optimization
        # n_calls: Total number of evaluations.
        # n_initial_points: How many random points to sample before using the model.
        result = gp_minimize(
            func=objective_function,
            dimensions=search_space,
            n_calls=50,
            n_initial_points=5,
            random_state=random_state,
            acq_func="EI",  # Expected Improvement acquisition function
        )

        end_time = time.time()
        print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")

        # --- Print and use the best results ---
        best_params = result.x
        best_score = -result.fun

        best_Q_diag = best_params[0:4]
        best_R_val = best_params[4]

        print("\n--- Best Parameters Found ---")
        print(
            f"Q_diag = [{best_Q_diag[0]:.4f}, {best_Q_diag[1]:.4f}, {best_Q_diag[2]:.4f}, {best_Q_diag[3]:.4f}]"
        )
        print(f"R      = [[{best_R_val:.4f}]]")
        print(f"Best Average Survival Time = {best_score:.2f} steps")

        # --- Demonstrate the tuned controller ---
        print("\n--- Demonstrating the Tuned LQR Controller ---")
        # demo_env = objective.make_eval_env("human")
        demo_env = objective.make_eval_env()

        print(f"Default pole length: {demo_env.unwrapped.length}")
        print(f"Default pole mass: {demo_env.unwrapped.masspole}")
        print(f"Default masscart: {demo_env.unwrapped.masscart}")
        print(f"Default gravity: {demo_env.unwrapped.gravity}")
        print(f"Default force_mag: {demo_env.unwrapped.force_mag}")

        # Re-calculate the final optimal K with the best parameters
        Q_final = np.diag(best_Q_diag)
        R_final = np.array([[best_R_val]])
        K_final = objective.lqr_controller(Q_final, R_final)

        for episode in range(5):
            state, _ = demo_env.reset()
            print(f"Starting Demonstration Episode {episode + 1}")

            for t in range(500):
                demo_env.render()
                error = state - np.array([0.0, 0.0, 0.0, 0.0])
                force = -(K_final @ error)[0]
                action = 1 if force > 0 else 0
                state, _, terminated, truncated, _ = demo_env.step(action)
                if terminated or truncated:
                    print(f"Episode finished after {t+1} steps.")
                    time.sleep(1)  # Pause before next episode
                    break
            else:
                print("Episode reached max steps successfully!")
                time.sleep(1)

        demo_env.close()
