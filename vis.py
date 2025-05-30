from stable_baselines3 import PPO
from acrobot_env import AcrobotMujocoEnv  # Update with your actual file/module name
# from acrobot_env_effort_v2 import AcrobotMujocoEnv  # Update the import
import time
import torch
# Load the trained model
model = PPO.load("ppo_acrobot_mujoco")
# model = PPO.load("ppo_acrobot_mujoco_effort_rew")
# model = PPO.load("ppo_acrobot_mujoco_effort-1")
# Create the environment with rendering enabled
env = AcrobotMujocoEnv(xml_path="physics_sim/acrobot.xml", render_mode=True)

# Run one episode
obs, _ = env.reset()
done, truncated = False, False
steps = 0
total_reward = 0
total_effort = 0
while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    time.sleep(0.01)
    steps+=1
    total_reward+=reward
    total_effort+=abs(action)
    env.render()
print(total_effort)
print(total_reward)
print(steps)
env.close()


