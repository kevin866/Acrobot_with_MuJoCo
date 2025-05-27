import torch
from acrobot_env import AcrobotMujocoEnv
from REINFORCE.model import PolicyNetwork  # Replace with actual file if needed
import time
import numpy as np

# Environment setup
env = AcrobotMujocoEnv(xml_path="physics_sim/acrobot.xml", render_mode=True)

# Define policy dimensions
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Recreate and load the trained policy
policy = PolicyNetwork(obs_dim, act_dim)
policy.load_state_dict(torch.load("reinforce_acrobot.pth"))
policy.eval()
obs, _ = env.reset()
done = False
truncated = False
steps = 0
total_reward = 0
total_effort = 0

while not (done or truncated):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        mean, std = policy(obs_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.mean.numpy()  # deterministic action

    obs, reward, done, truncated, _ = env.step(action)
    time.sleep(0.01)
    total_reward += reward
    total_effort += np.abs(action).sum()
    steps += 1
    env.render()

print("Total effort:", total_effort)
print("Total reward:", total_reward)
print("Steps:", steps)
env.close()
