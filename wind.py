import mujoco
import mujoco.viewer
import numpy as np

# Load model and data
model = mujoco.MjModel.from_xml_path("acrobot.xml")
data = mujoco.MjData(model)
link2_id = model.body("lower_arm").id
step_counter = 0
current_action = 0
# Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_counter+=1
        wind_force = np.random.normal(loc=0.0, scale=2.0, size=6)  # Gaussian wind disturbance
        # wind_force = np.array([np.random.normal(loc=0.0, scale=2.0), 0.0, 0.0, 0.0, 0.0, 0.0])  # constant wind in +x
        # print(wind_force)
        data.xfrc_applied[link2_id] = 1e-9

        # Random control signal to second joint
        # data.ctrl[0] = np.random.uniform(-1.0, 1.0)*50
        # # data.ctrl[0] = 0
        # if step_counter % 100 == 0:
        #     current_action = np.random.uniform(-5.0, 5.0)*2
        # data.ctrl[0] = current_action

        
        mujoco.mj_step(model, data)
        print(data.qpos)
        print("Damping:", model.dof_damping)

        viewer.sync()
