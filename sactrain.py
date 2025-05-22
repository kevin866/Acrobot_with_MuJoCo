import time
import os
from acrobot_env import AcrobotMujocoEnv  # Save your custom env as acrobot_env.py or inline it here
from SACagent import SACAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment and agent
env = AcrobotMujocoEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = SACAgent(state_dim, action_dim)

num_episodes = 500
max_timesteps = 1000
explore_steps = 10000  # Initial random steps before training
total_steps = 0

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for t in range(max_timesteps):
        if total_steps < explore_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        episode_reward += reward
        total_steps += 1

        if total_steps >= explore_steps:
            agent.update()

        if done:
            break

    print(f"Episode {episode + 1}, Reward: {episode_reward}, Steps: {total_steps}")

    if (episode + 1) % 10 == 0:
        eval_state, _ = env.reset()
        eval_reward = 0
        for _ in range(max_timesteps):
            eval_action = agent.select_action(eval_state, eval=True)
            eval_state, reward, eval_done, _, _ = env.step(eval_action)
            eval_reward += reward
            if eval_done:
                print("goal achieved!")
                break
        print(f"Evaluation Episode {episode + 1}, Reward: {eval_reward}")

env.close()
