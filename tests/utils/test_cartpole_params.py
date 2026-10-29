import numpy as np
import time
import jax
from jax import random
import gymnasium as gym
import imageio.v2 as imageio

from fedbo.data.objectives import CartpoleLQRObjective

if __name__ == "__main__":
    # random_state = 4
    random_state = 5
    np.random.seed(random_state)
    key = jax.random.key(random_state)
    key, subkey = random.split(key)
    objective = CartpoleLQRObjective(subkey)

    # demo_env = gym.make("CartPole-v1", render_mode="human")
    demo_env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Seed 6 agent 0 after 5 iteration best score -262.2
    # best_Q_diag = [1e-5, 0.17050979300786648, 1e-5, 1]
    # best_R_val = 1e-5
    # demo_env.unwrapped.gravity = 17.57995190211423
    # demo_env.unwrapped.masscart = 1.3153271820082502
    # demo_env.unwrapped.masspole = 0.15412910493932375
    # demo_env.unwrapped.length = 0.6646792024272248
    # demo_env.unwrapped.force_mag = 8.033352422394334

    # Seed 6 agent 10 after 5 iteration best score -424.9
    best_Q_diag = [1e-5, 1e-5, 0.8906189829039101, 0.6199110373534144]
    best_R_val = 1e-5
    demo_env.unwrapped.gravity = 18.358329228221812
    demo_env.unwrapped.masscart = 1.8939113817401994
    demo_env.unwrapped.masspole = 0.1988262018326103
    demo_env.unwrapped.length = 0.6998138633637006
    demo_env.unwrapped.force_mag = 12.618566031196234

    print(f"Q_diag = [{best_Q_diag[0]:.4f}, {best_Q_diag[1]:.4f}, {best_Q_diag[2]:.4f}, {best_Q_diag[3]:.4f}]")
    print(f"R      = [[{best_R_val:.4f}]]")

    print(f"pole length: {demo_env.unwrapped.length}")
    print(f"pole mass: {demo_env.unwrapped.masspole}")
    print(f"masscart: {demo_env.unwrapped.masscart}")
    print(f"gravity: {demo_env.unwrapped.gravity}")
    print(f"force_mag: {demo_env.unwrapped.force_mag}")

    # Re-calculate the final optimal K with the best parameters
    Q_final = np.diag(best_Q_diag)
    R_final = np.array([[best_R_val]])
    K_final = objective.lqr_controller(Q_final, R_final)

    all_frames = []
    for episode in range(5):
        key, subkey = random.split(key)
        seed = random.randint(subkey, shape=(), minval=0, maxval=1e6)
        state, _ = demo_env.reset(seed=int(seed))
        print(f"Starting Demonstration Episode {episode + 1}")

        for t in range(500):
            frame = demo_env.render()
            all_frames.append(frame)

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

    # Save the frames as a GIF
    imageio.mimsave("cartpole_demo.gif", all_frames, fps=30)  # Adjust fps as needed
    print("GIF saved as 'cartpole_demo.gif'")
