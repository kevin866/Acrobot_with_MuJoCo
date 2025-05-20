import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from acrobot_continuous import AcrobotContinuousEnv
# Your AcrobotContinuousEnv class here (use the exact one you provided)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable log std

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.rsample()  # reparameterization trick
        action_tanh = torch.tanh(action)
        log_prob = dist.log_prob(action) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action_tanh, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

def train_acrobot(env, actor, critic, actor_optimizer, critic_optimizer, num_episodes=500, gamma=0.99, entropy_coef=1e-3):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        entropies = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # shape: [1, state_dim]
            value = critic(state_tensor)
            action, log_prob = actor.get_action(state_tensor)
            entropy = -log_prob  # entropy estimate

            action_np = action.detach().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action_np)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)

        advantage = returns - values.detach()

        actor_loss = -(log_probs * advantage).mean() - entropy_coef * entropies.mean()
        critic_loss = nn.functional.mse_loss(values, returns)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Actor Loss: {actor_loss.item():.3f}, Critic Loss: {critic_loss.item():.3f}, Total Reward: {sum(rewards):.2f}")

    torch.save(actor.state_dict(), "acrobot_actor.pth")
    torch.save(critic.state_dict(), "acrobot_critic.pth")
    print("Models saved!")


if __name__ == "__main__":
    env = AcrobotContinuousEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    train_acrobot(env, actor, critic, actor_optimizer, critic_optimizer, num_episodes=300)
