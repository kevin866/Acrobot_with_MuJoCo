import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.mean_head = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # Learnable log std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        mean = self.mean_head(x)
        std = torch.exp(self.log_std)
        return mean, std

import gymnasium as gym
import numpy as np
import torch
def train_reinforce(env, policy, optimizer, num_episodes=1000, gamma=0.99):
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

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -torch.stack(log_probs) @ returns

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        all_rewards.append(episode_reward)

        writer.add_scalar("Reward/train", episode_reward, episode)

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward:.2f}")

    return all_rewards

def evaluate_policy(env, policy, num_episodes=10):
    policy.eval()
    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                mean, std = policy(obs)
                action = mean  # Use deterministic policy (mean)
            obs_new, reward, terminated, truncated, _ = env.step(action.numpy())
            done = terminated or truncated
            episode_reward += reward
            obs = torch.tensor(obs_new, dtype=torch.float32)

        print(f"Eval Episode {episode}, Total Reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average Eval Reward over {num_episodes} episodes: {avg_reward:.2f}")
def save_model(policy, path="reinforce_policy.pth"):
    torch.save(policy.state_dict(), path)

def load_model(policy, path="reinforce_policy.pth"):
    policy.load_state_dict(torch.load(path))
    policy.eval()

from acrobot_env_base import AcrobotMujocoEnv  # Update to your actual file name

env = AcrobotMujocoEnv()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

policy = PolicyNetwork(obs_dim, act_dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/acrobot_reinforce")

train_reinforce(env, policy, optimizer, num_episodes=500)
save_model(policy, "reinforce_acrobot.pth")
evaluate_policy(env, policy, num_episodes=10)

writer.close()
env.close()
