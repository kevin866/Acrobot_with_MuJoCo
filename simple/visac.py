import torch
from acn import Actor  # replace with your actual file/module



from acrobot_env import AcrobotMujocoEnv  # replace with your MuJoCo env

env = AcrobotMujocoEnv(xml_path="physics_sim/acrobot.xml", render_mode=True)
obs, _ = env.reset()
state_dim = env.observation_space.shape[0]  # e.g. 6 for AcrobotContinuousEnv
action_dim = env.action_space.shape[0]      # e.g. 1 for your continuous action
print(state_dim)
actor = Actor(state_dim, action_dim)
actor.load_state_dict(torch.load("simple/acrobot_actor.pth"))
actor.eval()
import numpy as np
import time
import torch

done, truncated = False, False
steps = 0

while not (done or truncated):
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # shape [1, state_dim]
    
    with torch.no_grad():
        mean, std = actor(obs_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = torch.tanh(dist.mean)  # deterministic action
    
    action_np = action.cpu().numpy()[0]
    
    obs, reward, done, truncated, _ = env.step(action_np)
    
    env.render()
    time.sleep(0.01)
    steps += 1

print("Episode steps:", steps)
env.close()
