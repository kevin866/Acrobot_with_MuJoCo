import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class AcrobotMujocoEnv(gym.Env):
    def __init__(self, xml_path="acrobot.xml", render_mode=False):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.link2_id = self.model.body("lower_arm").id

        self.max_steps = 1000
        self.step_count = 0
        self.render_mode = render_mode
        self.viewer = None
        self.target_height = 3  # meters, adjust based on link lengths


        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.model.nq + self.model.nv,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = [np.pi, 0.0]  # hanging
        self.data.qvel[:] = [0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        self.data.ctrl[0] = np.clip(action[0], -5.0, 5.0)

        wind_force = np.array([np.random.normal(0.0, 2.0), 0, 0, 0, 0, 0])
        self.data.xfrc_applied[self.link2_id] = wind_force

        mujoco.mj_step(self.model, self.data)

        if self.render_mode and self.viewer:
            self.viewer.sync()

        obs = self._get_obs()
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        tip_pos = self.data.site_xpos[site_id]
        tip_height = tip_pos[2]

        reward = tip_height  # baseline
        reward += -0.001 * np.sum(np.square(self.data.qvel))  # discourage flailing

        success = tip_height >= self.target_height
        terminated = success
        truncated = self.step_count >= self.max_steps

        return obs, reward, bool(terminated), bool(truncated), {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
