import numpy as np
import matplotlib.pyplot as plt
from dm_control import suite
from PIL import Image

env = suite.load(domain_name="acrobot", task_name="swingup")
env.reset()

qvel = env.physics.data.qvel
qvel[:] = np.zeros_like(qvel)

action_spec = env.action_spec()

def apply_wind_force():
    return np.random.uniform(low=-10.0, high=10.0, size=action_spec.shape)*10

plt.ion()
fig, ax = plt.subplots()

for _ in range(300):  # Simulate for ~5 seconds
    action = apply_wind_force()
    timestep = env.step(action)

    print("Wind force action applied:", action)
    # print("Current time step:", timestep)

    # Render and show the environment frame
    frame = env.physics.render(camera_id=0)
    ax.clear()
    ax.imshow(frame)
    plt.pause(0.01)
