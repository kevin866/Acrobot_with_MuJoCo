import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model and data
model = mujoco.MjModel.from_xml_path("acrobot2_act.xml")
data = mujoco.MjData(model)
link2_id = model.body("lower_arm").id

# Load control actions from the npy file
control_actions = np.load('best_actions.npy')  # shape (2048, 2)

# Initialize viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    step_counter = 0
    while viewer.is_running() and step_counter < control_actions.shape[0]:
        # Get the current control action (efforts for both joints)
        current_action = control_actions[step_counter]

        # Apply both control actions (one for each joint)
        data.ctrl[0] = current_action[0]  # first actuator
        data.ctrl[1] = current_action[1]  # second actuator

        # Step the simulation
        mujoco.mj_step(model, data)

        # Print the state (positions) of the joints
        print("Joint positions:", data.qpos)
        print("Damping:", model.dof_damping)

        # Increment the step counter
        step_counter += 1

        # Control the simulation speed
        time.sleep(0.01)

        # Sync viewer to show the updated simulation state
        viewer.sync()
