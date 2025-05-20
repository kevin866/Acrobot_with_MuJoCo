from gym.envs.registration import register

register(
    id="AcrobotContinuous-v0",
    entry_point="acrobot_continuous:AcrobotContinuousEnv",
)
