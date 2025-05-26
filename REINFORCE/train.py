import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
# Import your custom environment
from acrobot_env import AcrobotMujocoEnv  # Update to your actual file name

def train_reinforce(env, policy, optimizer, num_episodes=1000, gamma=0.99, log_dir="runs/reinforce_run"):
    writer = SummaryWriter(log_dir=log_dir)
    policy.train()
    all_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        log_probs = []
        rewards = []

        done = False
        while not done:
            mean, std = policy(obs)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            obs_new, reward, terminated, truncated, _ = env.step(action.detach().numpy())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            obs = torch.tensor(obs_new, dtype=torch.float32)

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        loss = -torch.stack(log_probs) @ returns
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        all_rewards.append(episode_reward)

        # 💾 Log to TensorBoard
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Loss/Episode", loss.item(), episode)

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward:.2f}")

    writer.close()
    return all_rewards
