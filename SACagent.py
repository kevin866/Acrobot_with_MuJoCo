import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random
import os

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state),
                torch.FloatTensor(action),
                torch.FloatTensor(reward).unsqueeze(1),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, alpha=0.2,
                 policy_lr=3e-4, q_lr=3e-4, buffer_size=1000000, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        self.policy = GaussianPolicy(state_dim, action_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)

        self.q1 = MLP(state_dim + action_dim, 1)
        self.q2 = MLP(state_dim + action_dim, 1)
        self.q1_target = MLP(state_dim + action_dim, 1)
        self.q2_target = MLP(state_dim + action_dim, 1)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=q_lr)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if eval:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), os.path.join(directory, f"{filename}_actor.pth"))
        torch.save(self.critic_1.state_dict(), os.path.join(directory, f"{filename}_critic1.pth"))
        torch.save(self.critic_2.state_dict(), os.path.join(directory, f"{filename}_critic2.pth"))
        torch.save(self.critic_target_1.state_dict(), os.path.join(directory, f"{filename}_target_critic1.pth"))
        torch.save(self.critic_target_2.state_dict(), os.path.join(directory, f"{filename}_target_critic2.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(directory, f"{filename}_actor_optim.pth"))
        torch.save(self.critic_1_optimizer.state_dict(), os.path.join(directory, f"{filename}_critic1_optim.pth"))
        torch.save(self.critic_2_optimizer.state_dict(), os.path.join(directory, f"{filename}_critic2_optim.pth"))


    def load(self, directory, filename):
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor.pth")))
        self.critic_1.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic1.pth")))
        self.critic_2.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic2.pth")))
        self.critic_target_1.load_state_dict(torch.load(os.path.join(directory, f"{filename}_target_critic1.pth")))
        self.critic_target_2.load_state_dict(torch.load(os.path.join(directory, f"{filename}_target_critic2.pth")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor_optim.pth")))
        self.critic_1_optimizer.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic1_optim.pth")))
        self.critic_2_optimizer.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic2_optim.pth")))


    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            q1_next = self.q1_target(torch.cat([next_state, next_action], 1))
            q2_next = self.q2_target(torch.cat([next_state, next_action], 1))
            q_target = reward + (1 - done) * self.gamma * (torch.min(q1_next, q2_next) - self.alpha * next_log_prob)

        q1_loss = F.mse_loss(self.q1(torch.cat([state, action], 1)), q_target)
        q2_loss = F.mse_loss(self.q2(torch.cat([state, action], 1)), q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_action, log_prob, _ = self.policy.sample(state)
        q1_new = self.q1(torch.cat([state, new_action], 1))
        q2_new = self.q2(torch.cat([state, new_action], 1))
        policy_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



