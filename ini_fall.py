import mujoco
import mujoco.viewer
import numpy as np

# Load your Acrobot model (assuming you have acrobot.xml)
model = mujoco.MjModel.from_xml_path("acrobot.xml")
data = mujoco.MjData(model)

# Set initial joint angles (in radians)
# qpos[0] = angle of joint 1, qpos[1] = angle of joint 2
data.qpos[:] = np.array([np.pi / 2, 0.0])  # top link at 90 degrees, second link hanging

# Set initial joint velocities to zero
data.qvel[:] = np.array([0.0, 0.0])
n = 0
# Run the simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if n==0:
            print(data.qpos)
            n+=1
        mujoco.mj_step(model, data)
        
        viewer.sync()
