import matplotlib.pyplot as plt
import numpy as np
from acn import Actor
from acrobot_continuous import AcrobotContinuousEnv
def plot_acrobot(state):
    theta1, theta2, _, _, _,_ = state
    # theta1 = theta1[0]
    # theta2 = theta2[0]

    # Link lengths
    l1, l2 = 1.0, 1.0
    
    # Compute (x,y) positions of joints
    x0, y0 = 0, 0
    x1 = l1 * np.cos(theta1 - np.pi/2)
    y1 = l1 * np.sin(theta1 - np.pi/2)
    x2 = x1 + l2 * np.cos(theta1 + theta2 - np.pi/2)
    y2 = y1 + l2 * np.sin(theta1 + theta2 - np.pi/2)

    plt.clf()
    plt.plot([x0, x1], [y0, y1], 'b-', linewidth=4)  # First link
    plt.plot([x1, x2], [y1, y2], 'r-', linewidth=4)  # Second link
    plt.plot(x0, y0, 'ko', markersize=10)             # Base joint
    plt.plot(x1, y1, 'ko', markersize=10)             # First joint
    plt.plot(x2, y2, 'ko', markersize=10)             # End effector
    
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal')
    plt.title("Acrobot Visualization")
    plt.pause(0.001)
import torch
# Usage example:
env = AcrobotContinuousEnv()
state = env.reset()
state_dim = 6  # e.g. 6 for AcrobotContinuousEnv
action_dim = 1
import time
actor = Actor(state_dim, action_dim)
actor.load_state_dict(torch.load("simple/acrobot_actor.pth"))
actor.eval()

done, truncated = False, False
steps = 0
while not (done or truncated):
    obs_tensor = torch.FloatTensor(state).unsqueeze(0).squeeze(-1)  # shape [1, state_dim]
    # print(obs_tensor.shape)
    with torch.no_grad():
        mean, std = actor(obs_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = torch.tanh(dist.mean)  # deterministic action
    
    action_np = action.cpu().numpy()[0]

    
    state, reward, done, _ = env.step(action_np)
    plot_acrobot(state)
    time.sleep(0.05)
    steps += 1
plt.show()
print("Episode steps:", steps)
