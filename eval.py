import numpy as np
from stable_baselines3 import PPO
from acrobot_env import AcrobotMujocoEnv  # Update the import
import gymnasium as gym

# Load trained model
model = PPO.load("ppo_acrobot_mujoco")

# Create environment
env = AcrobotMujocoEnv(xml_path="physics_sim/acrobot.xml", render_mode=False)

# Evaluation
num_episodes = 10
rewards = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done, truncated = False, False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    if truncated:
        print("not done")

env.close()

print(f"\nAverage reward over {num_episodes} episodes: {np.mean(rewards):.2f}")
