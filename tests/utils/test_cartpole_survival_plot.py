import numpy as np
import jax
from jax import random
import gymnasium as gym
import matplotlib.pyplot as plt
import RWTHColors

import importlib.resources

style_package = "fedbo.visualization.style"
style_file = "fedbo.mplstyle"
try:
    file_ref = importlib.resources.files(style_package).joinpath(style_file)

    with importlib.resources.as_file(file_ref) as path:
        plt.style.use(path)

except (ModuleNotFoundError, FileNotFoundError):
    print(f"Warning: Could not load the style '{style_file}' from '{style_package}'.")


from fedbo.data.objectives import CartpoleLQRObjective


def plot_survival(survival_data_list, max_steps, num_episodes):
    """
    Plots the average number of surviving episodes over time for multiple parameter sets,
    with a shaded region for uncertainty (standard deviation).

    Args:
        survival_data_list (list): A list of tuples, where each tuple is
                                     (label, all_survival_curves).
        max_steps (int): The maximum number of time steps.
        num_episodes (int): The number of episodes per seed.
    """
    plt.figure(figsize=(5.9, 3.5))
    time_steps = np.arange(max_steps)

    # Loop over each set of survival data
    for i, (label, all_survival_curves) in enumerate(survival_data_list):
        if not all_survival_curves:
            print(f"Warning: No survival data for '{label}'. Skipping.")
            continue

        survival_array = np.array(all_survival_curves)

        mean_survival = np.mean(survival_array, axis=0)
        std_survival = np.std(survival_array, axis=0)

        # Plot the mean survival curve
        plt.plot(time_steps, mean_survival, lw=2, label=f"{label}")

        # Plot the standard deviation band as a shaded area
        plt.fill_between(
            time_steps,
            mean_survival - std_survival,
            mean_survival + std_survival,
            alpha=0.3,
        )

    plt.xlabel("Step", fontsize="large")
    plt.ylabel("Number of Active Episodes", fontsize="large")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.ylim(10, num_episodes + 1)
    plt.xlim(0, 500)

    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(survival_data_list),
        frameon=False,
        fontsize="large",  
    )

    plt.savefig("survival_plot.pdf", bbox_inches="tight")
    plt.show()

    plt.close()


def run_simulation(param_set, num_seeds, num_episodes, max_steps):
    """
    Runs the CartPole LQR simulation for a given set of parameters over multiple seeds.

    Args:
        param_set (dict): A dictionary containing 'Q_diag', 'R_val', and 'env_params'.
        num_seeds (int): The number of simulation seeds to run.
        num_episodes (int): The number of episodes per seed.
        max_steps (int): The maximum duration of each episode.

    Returns:
        tuple: (all_seed_survival_curves, trajectories_for_last_seed)
    """
    print("\n" + "=" * 50)
    print(f"--- Running Simulation for Parameter Set ---")
    print(f"Q_diag: {param_set['Q_diag']}")
    print(f"R_val:  {param_set['R_val']}")
    print(f"Env:    {param_set['env_params']}")
    print("=" * 50)

    # --- LQR and Environment Parameters ---
    base_key = jax.random.key(0)  
    objective = CartpoleLQRObjective(base_key)

    Q_final = np.diag(param_set["Q_diag"])
    R_final = np.array([[param_set["R_val"]]])
    K_final = objective.lqr_controller(Q_final, R_final)

    demo_env = gym.make("CartPole-v1")

    # Set environment parameters
    env_p = param_set["env_params"]
    demo_env.unwrapped.gravity = env_p["gravity"]
    demo_env.unwrapped.masscart = env_p["masscart"]
    demo_env.unwrapped.masspole = env_p["masspole"]
    demo_env.unwrapped.length = env_p["length"]
    demo_env.unwrapped.force_mag = env_p["force_mag"]

    all_seed_survival_curves = []
    trajectories_for_last_seed = None

    # --- Main Loop for Seeds ---
    for i in range(num_seeds):
        current_seed = i
        if (i + 1) % 5 == 0 or i == 0:  # Print progress update
            print(f"   ... processing seed {current_seed + 1}/{num_seeds}")

        run_key = jax.random.key(current_seed)
        all_trajectories = []

        for episode in range(num_episodes):
            run_key, subkey = random.split(run_key)
            episode_seed = random.randint(subkey, shape=(), minval=0, maxval=int(1e6))
            state, _ = demo_env.reset(seed=int(episode_seed))

            current_trajectory = []
            for t in range(max_steps):
                current_trajectory.append(state)
                error = state - np.array([0.0, 0.0, 0.0, 0.0])
                force = -(K_final @ error)[0]
                action = 1 if force > 0 else 0
                state, _, terminated, truncated, _ = demo_env.step(action)
                if terminated or truncated:
                    break
            all_trajectories.append(np.array(current_trajectory))

        # --- Data Collection for Survival Plot ---
        survival_times = [len(traj) for traj in all_trajectories]
        survivors = [np.sum(np.array(survival_times) >= t) for t in range(max_steps)]
        all_seed_survival_curves.append(survivors)

        if i == num_seeds - 1:
            trajectories_for_last_seed = all_trajectories

    demo_env.close()
    print("--- Simulation for set complete. ---")
    return all_seed_survival_curves, trajectories_for_last_seed


if __name__ == "__main__":
    # --- Configuration ---
    num_seeds = 30
    num_episodes = 50
    max_steps = 500

    # --- Parameter Set 1 ---
    params_1 = {
        "Q_diag": [1e-5, 0.17050979300786648, 1e-5, 1],
        "R_val": 1e-5,
        "env_params": {
            "gravity": 17.57995190211423,
            "masscart": 1.3153271820082502,
            "masspole": 0.15412910493932375,
            "length": 0.6646792024272248,
            "force_mag": 8.033352422394334,
        },
    }

    # --- Parameter Set 2 ---
    params_2 = {
        "Q_diag": [1e-5, 1e-5, 0.792010225723113, 0.0484487740108645],
        "R_val": 1e-5,
        "env_params": {
            "gravity": 18.358329228221812,
            "masscart": 1.8939113817401994,
            "masspole": 0.1988262018326103,
            "length": 0.6998138633637006,
            "force_mag": 12.618566031196234,
        },
    }

    # --- Run Simulations ---

    # Run 1
    survival_curves_1, _ = run_simulation(params_1, num_seeds, num_episodes, max_steps)

    # Run 2
    survival_curves_2, trajectories_for_plot = run_simulation(params_2, num_seeds, num_episodes, max_steps)

    # --- Plotting Results ---
    print("\nGenerating average survival plot for both sets...")
    survival_data_to_plot = [("Cold-Start", survival_curves_1), ("Warm-Start", survival_curves_2)]
    plot_survival(survival_data_to_plot, max_steps=max_steps, num_episodes=num_episodes)
