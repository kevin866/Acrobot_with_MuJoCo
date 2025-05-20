from stable_baselines3 import PPO
from acrobot_env import AcrobotMujocoEnv  # Update with your actual file/module name
import time
# Load the trained model
model = PPO.load("ppo_acrobot_mujoco")

# Create the environment with rendering enabled
env = AcrobotMujocoEnv(xml_path="physics_sim/acrobot.xml", render_mode=True)

# Run one episode
obs, _ = env.reset()
done, truncated = False, False
steps = 0
while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    time.sleep(0.01)
    steps+=1
    env.render()
print(steps)
env.close()
