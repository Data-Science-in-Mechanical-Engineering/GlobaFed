import jax.numpy as jnp
import gymnasium as gym
import numpy as np
import time
from jax import random
from fedbo.data.objectives.cartpole_lqr_objective import CartpoleLQRObjective


if __name__ == "__main__":
    key = random.key(1)
    cartpole_obj = CartpoleLQRObjective(key)
    x = jnp.array([[25.2080, 1.7885, 0.0116, 514.0540, 0.6553]])
    cost = cartpole_obj(x)
    print(cost)

    # --- Demonstrate the tuned controller ---
    print("\n--- Demonstrating LQR Controller ---")
    # demo_env = cartpole_obj.make_eval_env()
    demo_env = cartpole_obj.make_eval_env(render_mode="human")

    print(f"Gravity: {demo_env.unwrapped.gravity}")
    print(f"Masscart: {demo_env.unwrapped.masscart}")
    print(f"Pole mass: {demo_env.unwrapped.masspole}")
    print(f"Pole length: {demo_env.unwrapped.length}")
    print(f"Force Magnitude: {demo_env.unwrapped.force_mag}")

    Q_final = np.diag(x[0, :4])
    R_final = np.array([[x[0, -1]]])
    K_final = cartpole_obj.lqr_controller(Q_final, R_final)

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
