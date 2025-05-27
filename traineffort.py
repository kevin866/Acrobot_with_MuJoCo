import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Import your custom environment
from acrobot_env_effort import AcrobotMujocoEnv  # Update to your actual file name

# Wrap it into a Gym-compatible function
def make_env():
    return AcrobotMujocoEnv(xml_path="physics_sim/acrobot.xml", render_mode=False)

# Check environment
env = make_env()
check_env(env)

# Vectorize environment
vec_env = DummyVecEnv([make_env])

# Define model
model = PPO(
    "MlpPolicy",
    vec_env,
    policy_kwargs=dict(net_arch=[64, 128, 256, 128, 64]),  # Your custom size
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.9,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.1,
    verbose=1,
    tensorboard_log="./ppo_acrobot_tensorboard/"
)

# Train model
model.learn(total_timesteps=900_000)

# Save model
model.save("ppo_acrobot_mujoco_effort")

# Optional: Test the trained model
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    # env.render()
    if done or truncated:
        obs, _ = env.reset()

env.close()
