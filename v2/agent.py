from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from acrobot_env import AcrobotMujocoEnv
# Create environment
env = AcrobotMujocoEnv("physics_sim/acrobot.xml")
check_env(env)  # Optional: check if your env is valid

# Train PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

# Save model
model.save("ppo_acrobot_mujoco")
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    print(obs)
    if done:
        break



