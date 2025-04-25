# import gym
import numpy as np
import os
import imageio

from dm_control import suite
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
import gymnasium as gym
import numpy as np
from dm_control import suite
import gymnasium as gym
import numpy as np
from dm_control import suite

class DMControlWrapper(gym.Env):  # Use gymnasium.Env here
    def __init__(self, domain_name='acrobot', task_name='swingup'):
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        self.action_space = gym.spaces.Box(low=self.env.action_spec().minimum,
                                           high=self.env.action_spec().maximum,
                                           dtype=np.float32)

        obs_spec = self.env.observation_spec()
        self.obs_keys = list(obs_spec.keys())
        obs_dim = sum(np.prod(spec.shape) for spec in obs_spec.values())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.time_step = None

    def reset(self, **kwargs):  # Ensure reset matches Gymnasium's signature
        self.time_step = self.env.reset()
        # Return only the observation in the correct format
        return self._flatten_obs(self.time_step.observation), {}

    def step(self, action):
        self.time_step = self.env.step(action)
        obs = self._flatten_obs(self.time_step.observation)
        reward = self.time_step.reward or 0.0
        done = self.time_step.last()
        return obs, reward, done, False, {}

    def _flatten_obs(self, obs_dict):
        return np.concatenate([np.array(obs_dict[k]).ravel() for k in self.obs_keys])



# 2. Environment and Logging Setup
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Replace gym with gymnasium
env = DummyVecEnv([lambda: Monitor(DMControlWrapper())])


# Optional: Configure tensorboard logging
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# 3. Define and Train PPO Model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.set_logger(new_logger)

# Callback to evaluate every 10k steps
eval_env = DummyVecEnv([lambda: DMControlWrapper()])
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=10_000,
                             deterministic=True, render=False)

model.learn(total_timesteps=100_000, callback=eval_callback)
model.save(os.path.join(log_dir, "ppo_acrobot_dm"))

# 4. Save a video of trained agent
def record_video(model, path="acrobot_trained.mp4", steps=500):
    env = DMControlWrapper()
    obs, info = env.reset()  # Gym-style reset (tuple)
    frames = []

    for _ in range(steps):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        frame = env.env.physics.render(height=480, width=640, camera_id=0)
        frames.append(frame)
        if done:
            obs, info = env.reset()  # Gym-style reset (tuple)


    imageio.mimsave(path, frames, fps=30)
    print(f"🎥 Saved video to {path}")

record_video(model)

print("✅ Training complete. Check TensorBoard with:")
print(f"tensorboard --logdir {log_dir}")
