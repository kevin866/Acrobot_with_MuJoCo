import time
from stable_baselines3 import PPO
from acrobot_env import AcrobotMujocoEnv
import mujoco

env = AcrobotMujocoEnv("acrobot.xml", render_mode=True)
model = PPO.load("ppo_acrobot_mujoco")

obs, _ = env.reset()
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    env.viewer = viewer  # attach viewer for syncing
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        time.sleep(0.01)  # slow it down for viewing
        if done:
            break
env.close()
