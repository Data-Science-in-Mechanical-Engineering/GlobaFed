import numpy as np
import time
import jax
from jax import random
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from fedbo.data.objectives import CartpoleLQRObjective


def plot_trajectories(trajectories, mean_trajectory, std_trajectory, seed_number):
    """Plots the average trajectory with standard deviation for a single seed run."""
    time_steps = np.arange(mean_trajectory.shape[0])
    state_labels = ["Cart Position (x)", "Cart Velocity (ẋ)", "Pole Angle (θ)", "Pole Angular Velocity (θ̇)"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Average Trajectory of State Variables (Seed {seed_number})", fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        mean = mean_trajectory[:, i]
        std = std_trajectory[:, i]

        # Plot the mean
        ax.plot(time_steps, mean, color="royalblue", lw=2, label="Mean")

        # Plot the standard deviation band
        ax.fill_between(time_steps, mean - std, mean + std, color="cornflowerblue", alpha=0.3, label="Std. Deviation")

        ax.set_title(state_labels[i])
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        if i >= 2:
            ax.set_xlabel("Time Step")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_survival(all_survival_curves, max_steps, num_episodes):
    """
    Plots the average number of surviving episodes over time across multiple seeds,
    with a shaded region for uncertainty (standard deviation).
    """
    # Convert list of lists to a NumPy array for easier calculations
    survival_array = np.array(all_survival_curves)

    mean_survival = np.mean(survival_array, axis=0)
    std_survival = np.std(survival_array, axis=0)
    time_steps = np.arange(max_steps)

    plt.figure(figsize=(10, 6))

    # Plot the mean survival curve
    plt.plot(time_steps, mean_survival, lw=2, color="darkgreen", label="Mean Survival")

    # Plot the standard deviation band as a shaded area
    plt.fill_between(
        time_steps,
        mean_survival - std_survival,
        mean_survival + std_survival,
        color="mediumseagreen",
        alpha=0.3,
        label="Std. Deviation",
    )

    plt.title(f"Average Survival Plot of Episodes over {len(all_survival_curves)} Seeds", fontsize=16)
    plt.xlabel("Time Step")
    plt.ylabel("Number of Active Episodes")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, num_episodes + 1)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # --- Configuration ---
    num_seeds = 30
    num_episodes = 50
    max_steps = 500

    best_Q_diag = [1e-5, 0.17050979300786648, 1e-5, 1]
    best_R_val = 1e-5

    # --- LQR and Environment Parameters (fixed for all runs) ---
    # best_Q_diag = [1e-5, 1e-5, 0.792010225723113, 0.0484487740108645]
    # best_R_val = 1e-5

    # print(f"Q_diag = [{best_Q_diag[0]:.4f}, {best_Q_diag[1]:.4f}, {best_Q_diag[2]:.4f}, {best_Q_diag[3]:.4f}]")
    # print(f"R      = [[{best_R_val:.4f}]]")

    # This objective object can be reused for all seeds
    base_key = jax.random.key(0)
    objective = CartpoleLQRObjective(base_key)

    Q_final = np.diag(best_Q_diag)
    R_final = np.array([[best_R_val]])
    K_final = objective.lqr_controller(Q_final, R_final)

    demo_env = gym.make("CartPole-v1")  # No render mode needed here

    demo_env.unwrapped.gravity = 17.57995190211423
    demo_env.unwrapped.masscart = 1.3153271820082502
    demo_env.unwrapped.masspole = 0.15412910493932375
    demo_env.unwrapped.length = 0.6646792024272248
    demo_env.unwrapped.force_mag = 8.033352422394334

    # Set environment parameters
    # demo_env.unwrapped.gravity = 18.358329228221812
    # demo_env.unwrapped.masscart = 1.8939113817401994
    # demo_env.unwrapped.masspole = 0.1988262018326103
    # demo_env.unwrapped.length = 0.6998138633637006
    # demo_env.unwrapped.force_mag = 12.618566031196234

    all_seed_survival_curves = []
    trajectories_for_last_seed = None  # To store for the trajectory plot

    # --- Main Loop for 30 Seeds ---
    for i in range(num_seeds):
        current_seed = i
        print(f"\n--- Running Simulation for Seed {current_seed + 1}/{num_seeds} ---")

        # We use a different jax key for each simulation run for better randomness
        run_key = jax.random.key(current_seed)

        all_trajectories = []
        for episode in range(num_episodes):
            run_key, subkey = random.split(run_key)
            # Use a unique seed for each episode reset
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

        # Save the trajectories of the last seed to generate the state plot
        if i == num_seeds - 1:
            trajectories_for_last_seed = all_trajectories

    demo_env.close()

    # --- Plotting Results ---

    # 1. Plot the average survival curve with uncertainty over 30 seeds
    print("\nGenerating average survival plot...")
    plot_survival(all_seed_survival_curves, max_steps=max_steps, num_episodes=num_episodes)

    # 2. Plot the state trajectories for the last seed as a representative example
    print(f"Generating trajectory plot for the last run (Seed {num_seeds})...")
    # Pad trajectories from the last run to calculate mean and std
    max_len = max(len(traj) for traj in trajectories_for_last_seed)
    num_states = trajectories_for_last_seed[0].shape[1]
    padded_trajectories = np.full((len(trajectories_for_last_seed), max_len, num_states), np.nan)
    for i, traj in enumerate(trajectories_for_last_seed):
        padded_trajectories[i, : len(traj), :] = traj

    mean_trajectory = np.nanmean(padded_trajectories, axis=0)
    std_trajectory = np.nanstd(padded_trajectories, axis=0)

    plot_trajectories(padded_trajectories, mean_trajectory, std_trajectory, seed_number=num_seeds)
