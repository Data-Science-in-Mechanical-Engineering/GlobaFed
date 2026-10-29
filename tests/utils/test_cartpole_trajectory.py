import numpy as np
import time
import jax
from jax import random
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from fedbo.data.objectives import CartpoleLQRObjective


def plot_trajectories(trajectories, mean_trajectory, std_trajectory):
    """Plots the average trajectory with standard deviation."""
    time_steps = np.arange(mean_trajectory.shape[0])
    state_labels = ["Cart Position (x)", "Cart Velocity (ẋ)", "Pole Angle (θ)", "Pole Angular Velocity (θ̇)"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Average Trajectory of CartPole State Variables", fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        mean = mean_trajectory[:, i]
        std = std_trajectory[:, i]

        # Plot individual trajectories for context (optional, can be noisy)
        # for traj in trajectories:
        #     ax.plot(traj[:, i], color='gray', alpha=0.2)

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


def plot_survival(all_trajectories, max_steps):
    """Plots the number of surviving episodes over time."""
    num_episodes = len(all_trajectories)
    survival_times = [len(traj) for traj in all_trajectories]

    survivors = [np.sum(np.array(survival_times) >= t) for t in range(max_steps)]

    plt.figure(figsize=(10, 6))
    plt.plot(range(max_steps), survivors, lw=2, color="darkgreen")
    plt.title("Survival Plot of Episodes", fontsize=16)
    plt.xlabel("Time Step")
    plt.ylabel("Number of Active Episodes")
    plt.grid(True, linestyle="--", alpha=0.6)
    # Set y-axis limits from 0 to the total number of episodes
    plt.ylim(0, num_episodes + 1)
    plt.show()


if __name__ == "__main__":
    random_state = 5
    np.random.seed(random_state)
    key = jax.random.key(random_state)
    key, subkey = random.split(key)
    objective = CartpoleLQRObjective(subkey)

    demo_env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Seed 6 agent 0 after 5 iteration best score -262.2
    # best_Q_diag = [1e-5, 0.17050979300786648, 1e-5, 1]
    # best_R_val = 1e-5
    # demo_env.unwrapped.gravity = 17.57995190211423
    # demo_env.unwrapped.masscart = 1.3153271820082502
    # demo_env.unwrapped.masspole = 0.15412910493932375
    # demo_env.unwrapped.length = 0.6646792024272248
    # demo_env.unwrapped.force_mag = 8.033352422394334

    # After 10 iterations -339.8
    # best_Q_diag = [1e-5, 0.34597389646194693, 0.4233817403573931, 1]
    # best_R_val = 1e-5

    # Seed 6 agent 10 after 5 iteration best score -424.9
    # best_Q_diag = [1e-5, 1e-5, 0.8906189829039101, 0.6199110373534144]
    # best_R_val = 1e-5
    demo_env.unwrapped.gravity = 18.358329228221812
    demo_env.unwrapped.masscart = 1.8939113817401994
    demo_env.unwrapped.masspole = 0.1988262018326103
    demo_env.unwrapped.length = 0.6998138633637006
    demo_env.unwrapped.force_mag = 12.618566031196234

    # After 5 iterations 495
    best_Q_diag = [1e-5, 1e-5, 0.792010225723113, 0.0484487740108645]
    best_R_val = 1e-5

    # After 5 iterations -496.6
    # best_Q_diag = [1e-5, 1e-5, 0.768129096574506, 0.08969632238959976]
    # best_R_val = 1e-5

    print(f"Q_diag = [{best_Q_diag[0]:.4f}, {best_Q_diag[1]:.4f}, {best_Q_diag[2]:.4f}, {best_Q_diag[3]:.4f}]")
    print(f"R      = [[{best_R_val:.4f}]]")

    # Re-calculate the final optimal K with the best parameters
    Q_final = np.diag(best_Q_diag)
    R_final = np.array([[best_R_val]])
    K_final = objective.lqr_controller(Q_final, R_final)

    all_frames = []
    all_trajectories = []
    num_episodes = 50

    for episode in range(num_episodes):
        key, subkey = random.split(key)
        seed = random.randint(subkey, shape=(), minval=0, maxval=int(1e6))
        state, _ = demo_env.reset(seed=int(seed))
        print(f"Starting Episode {episode + 1}/{num_episodes}")

        current_trajectory = []  # Store states for this episode
        for t in range(500):
            # Save frame for GIF only for the first episode
            if episode == 0:
                frame = demo_env.render()
                all_frames.append(frame)

            current_trajectory.append(state)

            error = state - np.array([0.0, 0.0, 0.0, 0.0])
            force = -(K_final @ error)[0]
            action = 1 if force > 0 else 0
            state, _, terminated, truncated, _ = demo_env.step(action)
            if terminated or truncated:
                break

        all_trajectories.append(np.array(current_trajectory))

    demo_env.close()

    # Find the length of the longest episode
    max_len = max(len(traj) for traj in all_trajectories)
    num_states = all_trajectories[0].shape[1]

    # Create an array filled with NaNs and copy trajectories into it
    padded_trajectories = np.full((len(all_trajectories), max_len, num_states), np.nan)
    for i, traj in enumerate(all_trajectories):
        padded_trajectories[i, : len(traj), :] = traj

    # Calculate mean and std, ignoring NaNs
    mean_trajectory = np.nanmean(padded_trajectories, axis=0)
    std_trajectory = np.nanstd(padded_trajectories, axis=0)

    # Plot the results
    plot_trajectories(padded_trajectories, mean_trajectory, std_trajectory)
    plot_survival(all_trajectories, max_steps=500)
