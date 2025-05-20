import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

# Import your custom environment
import acrobot_continuous  # this is the file you created

# Register the environment (skip if already registered in acrobot_continuous.py)
from gym.envs.registration import register

try:
    register(
        id='AcrobotContinuous-v0',
        entry_point='acrobot_continuous:AcrobotContinuousEnv',
    )
except:
    # Already registered
    pass

# Create the environment
env = gym.make("AcrobotContinuous-v0")

# Optional: check for environment compliance with Gym API
# check_env(env)

# Create the model (Soft Actor-Critic)
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_acrobot_tensorboard")

# Train the model
model.learn(total_timesteps=100_000)

# Save the model
model.save("sac_acrobot_continuous")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
