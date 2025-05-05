import numpy as np
from dm_control import suite
from dm_control import viewer

# Load Acrobot swing-up task
env = suite.load(domain_name="acrobot", task_name="swingup")

# Set initial state to be still (zero velocity and neutral position)
# The state is a tuple of observation and the action space is part of the env.state.
# We can manually set the state to ensure that the robot starts still.

# Reset the environment to start from the beginning (still state)
env.reset()

# Access the joint positions and velocities
qpos = env.physics.data.qpos  # Joint positions
qvel = env.physics.data.qvel  # Joint velocities

# Set joint velocities to zero to start the Acrobot still
qvel[:] = np.zeros_like(qvel)  # Zero out joint velocities for a still start

# Launch the interactive viewer
viewer.launch(env)

# Get the current action space from the environment
action_spec = env.action_spec()

# Function to apply random wind force
def apply_wind_force():
    # Generate random action (force between -1.0 and 1.0 for each joint)
    random_action = np.random.uniform(low=-1.0, high=1.0, size=action_spec.shape)

    # Apply a random force to simulate wind (e.g., on the second joint)
    random_action[1] = np.random.uniform(low=-1.0, high=1.0)  # Random wind force for the upper body

    return random_action

# Run the simulation loop
while True:
    # Apply the random wind force after the initial still state
    wind_force_action = apply_wind_force()

    # Apply the action to the environment and step forward
    time_step = env.step(wind_force_action)

    # Optionally, print information to see progress (e.g., angles, velocities)
    print("Wind force action applied:", wind_force_action)
    print("Current time step:", time_step)

    # Visualize the environment in the interactive viewer
    viewer.render()
