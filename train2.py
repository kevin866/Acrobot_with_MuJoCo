
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Import your custom environment
from acrobot_env import AcrobotMujocoEnv  # Update to your actual file name

# Wrap it into a Gym-compatible function
def make_env():
    return AcrobotMujocoEnv(xml_path="physics_sim/acrobot.xml", render_mode=False)

# Check environment
env = make_env()
check_env(env)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# Parallelize environments for better batch variance reduction
vec_env = SubprocVecEnv([make_env for _ in range(4)])

# Normalize observations and rewards
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=1e-4,  # reduce LR
    n_steps=1024,        # adjust to episode length
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,      # tweak entropy coefficient
    verbose=1,
    tensorboard_log="./ppo_acrobot_tensorboard/"
)

model.learn(total_timesteps=1_000_000)  # more training time

# Save model
model.save("ppo_acrobot_mujoco")

# Optional: Test the trained model
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()

env.close()
