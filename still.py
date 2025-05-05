import numpy as np
from dm_control import suite
from dm_control import viewer

# Load Acrobot swingup task
env = suite.load(domain_name="acrobot", task_name="swingup")

# Reset environment and manually override the physics state
env.reset()

# Set both joint positions and velocities
# [theta1, theta2], values like 0.0 = downward, π = upward
env.physics.named.data.qpos[:] = [0.0, 0.0]  # both links hanging down
env.physics.named.data.qvel[:] = [0.0, 0.0]  # no motion

# Important: update internal physics to reflect state change
env.physics.forward()

# Confirm state before launching viewer
print("Initial qpos:", env.physics.named.data.qpos[:])
print("Initial qvel:", env.physics.named.data.qvel[:])

# Launch viewer with your policy
viewer.launch(env)




