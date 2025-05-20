import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from acrobot_env import AcrobotMujocoEnv  # Replace with the actual file name where AcrobotMujocoEnv is defined

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
ENTROPY_BETA = 0.001
EPISODES = 1000
MAX_STEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

def train():
    env = AcrobotMujocoEnv()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim).to(device)

    actor_optim = optim.Adam(actor.parameters(), lr=LR)
    critic_optim = optim.Adam(critic.parameters(), lr=LR)

    for episode in range(EPISODES):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        total_reward = 0

        for step in range(MAX_STEPS):
            mu = actor(obs)
            dist = torch.distributions.Normal(mu, 1.0)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().detach().numpy())
            done = terminated or truncated
            total_reward += reward

            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            done_tensor = torch.tensor([done], dtype=torch.float32, device=device)

            # Compute advantage
            value = critic(obs)
            next_value = critic(next_obs_tensor)
            target = reward_tensor + (1 - done_tensor) * GAMMA * next_value
            advantage = target - value

            # Actor loss
            actor_loss = -(log_prob * advantage.detach()) - ENTROPY_BETA * dist.entropy().sum()

            # Critic loss
            critic_loss = advantage.pow(2).mean()

            # Backpropagation
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            if done:
                break

            obs = next_obs_tensor

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    # Save actor model
    torch.save(actor.state_dict(), "acrobot_actor.pth")
    print("Model saved as 'acrobot_actor.pth'")

if __name__ == "__main__":
    train()
