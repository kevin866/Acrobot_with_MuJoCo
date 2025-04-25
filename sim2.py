import numpy as np
from dm_control import suite

# Load the environment
env = suite.load(domain_name="acrobot", task_name="swingup")

# Reset the environment
time_step = env.reset()

for step in range(500):
    # Sample a random action (between -1 and 1)
    action = np.random.uniform(low=-1.0, high=1.0, size=env.action_spec().shape)

    # Step the simulation
    time_step = env.step(action)

    # Access observations and reward
    obs = time_step.observation
    reward = time_step.reward
    done = time_step.last()
    
    print(f"Step: {step}, Reward: {reward}, Done: {done}")
