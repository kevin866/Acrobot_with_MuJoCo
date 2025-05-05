import mujoco
import mujoco.viewer
import numpy as np

# Load model and data
model = mujoco.MjModel.from_xml_path("acrobot.xml")
data = mujoco.MjData(model)
link2_id = model.body("link2").id

# Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        wind_force = np.random.normal(loc=0.0, scale=20.0, size=6)  # Gaussian wind disturbance
        wind_force = np.array([1e9, 0.0, 0.0, 0.0, 0.0, 0.0])  # constant wind in +x
        data.xfrc_applied[link2_id] = wind_force

        # Random control signal to second joint
        # data.ctrl[0] = np.random.uniform(-1.0, 1.0)*10
        data.ctrl[0] = 1e3
        
        mujoco.mj_step(model, data)
        print(data.qpos)
        viewer.sync()
