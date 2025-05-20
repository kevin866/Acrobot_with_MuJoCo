import gym
import acrobot_continuous  # your Python file

env = gym.make("AcrobotContinuous-v0")
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break
