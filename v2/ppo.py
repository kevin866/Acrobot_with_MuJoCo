import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
import os

class AcrobotEnv(MujocoEnv):
    def __init__(self, **kwargs):
        model_path = os.path.abspath("acrobot2_act.xml")
        super().__init__(
            model_path=model_path,  # path to the saved XML
            frame_skip=4,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            **kwargs
        )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = -np.linalg.norm(self.data.site("tip").xpos - np.array([0, 0, 4]))
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Learnable log standard deviation for actions
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mu = self.actor(obs)
        std = self.log_std.exp().expand_as(mu)
        value = self.critic(obs)
        return mu, std, value


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    values = values + [0]  # bootstrap
    gae = 0
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        returns.insert(0, gae + values[i])
    return returns


def train():
    # from your_script import AcrobotEnv  # or replace with your class directly

    env = AcrobotEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    best_reward = -np.inf
    best_actions = []
    max_episodes = 1000
    batch_size = 2048
    update_epochs = 10
    clip_epsilon = 0.2

    for episode in range(max_episodes):
        obs, _ = env.reset()
        obs_buffer = []
        act_buffer = []
        logp_buffer = []
        rew_buffer = []
        val_buffer = []
        done_buffer = []
        ep_reward = 0
        episode_actions = []

        for _ in range(batch_size):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            mu, std, value = model(obs_tensor)
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated

            obs_buffer.append(obs)
            act_buffer.append(action.cpu().numpy())
            logp_buffer.append(log_prob.item())
            rew_buffer.append(reward)
            val_buffer.append(value.item())
            done_buffer.append(done)
            episode_actions.append(action.cpu().numpy())

            obs = next_obs
            ep_reward += reward
            if done:
                obs, _ = env.reset()

        returns = compute_gae(rew_buffer, val_buffer, done_buffer)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        obs_tensor = torch.tensor(obs_buffer, dtype=torch.float32).to(device)
        act_tensor = torch.tensor(act_buffer, dtype=torch.float32).to(device)
        logp_old_tensor = torch.tensor(logp_buffer, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, _, values = model(obs_tensor)
            adv = returns - values.squeeze()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(update_epochs):
            mu, std, value = model(obs_tensor)
            dist = Normal(mu, std)
            logp = dist.log_prob(act_tensor).sum(axis=1)
            ratio = torch.exp(logp - logp_old_tensor)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - value.squeeze()).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_actions = episode_actions.copy()
            np.save("best_actions.npy", np.array(best_actions))

        print(f"Episode {episode}, Reward: {ep_reward:.2f}, Best: {best_reward:.2f}")

    print(f"Training complete. Best reward: {best_reward:.2f}. Actions saved to 'best_actions.npy'")


if __name__ == "__main__":
    train()
