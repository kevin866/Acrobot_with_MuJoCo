import mujoco
import numpy as np
import imageio
import time

model = mujoco.MjModel.from_xml_path("acrobot.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

link2_id = model.body("lower_arm").id

frames = []
data = mujoco.MjData(model)
data.qpos[:] = [np.pi, 0.0]  # or [0.0, 0.0] depending on model
mujoco.mj_forward(model, data)

for _ in range(500):  # Adjust number of frames as needed
    # Apply wind force
    wind_force = np.array([np.random.normal(loc=0.0, scale=1.0), 0.0, 0.0, 0.0, 0.0, 0.0])
    data.xfrc_applied[link2_id] = wind_force

    mujoco.mj_step(model, data)

    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)

    # Slow down the sim
    time.sleep(0.02)

# Save video
imageio.mimsave("simulation.mp4", frames, fps=30)  # Change fps for slower/faster video
