import numpy as np
from stable_baselines3 import PPO
# from acrobot_env_effort_v2 import AcrobotMujocoEnv  # Update the import
# from acrobot_env import AcrobotMujocoEnv  # Update the import
from acrobot_env_effort import AcrobotMujocoEnv  # Update the import

import gymnasium as gym
import time
# Load trained model
model = PPO.load("ppo_acrobot_mujoco")
# model = PPO.load("ppo_acrobot_mujoco_effort")
# model = PPO.load("ppo_acrobot_mujoco_effort_rew")

# model = PPO.load("ppo_acrobot_mujoco_effort-1")

# Create environment
env = AcrobotMujocoEnv(xml_path="physics_sim/acrobot.xml", render_mode=False)

# Evaluation
goals = 0
num_episodes = 100
rewards = []
total_action = 0
for episode in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done, truncated = False, False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        total_action+=abs(action)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    # print(total_action)
    if done:
        goals+=1
        print("Goal reached!")

env.close()
avg_action = total_action/num_episodes
print(f"\nAverage reward over {num_episodes} episodes: {np.mean(rewards):.2f}")
print("\nAverage action effot{}".format(avg_action))
print(goals)