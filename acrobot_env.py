import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class AcrobotMujocoEnv(gym.Env):
    def __init__(self, xml_path="physics_sim/acrobot.xml", render_mode=False):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.link2_id = self.model.body("lower_arm").id
        self.seed = 0
        self.max_steps = 2000
        self.step_count = 0
        self.render_mode = render_mode
        self.viewer = None
        self.target_height = 3  # meters, adjust based on link lengths
        self.current_episode = 0
        self.total_episodes = 500

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
        self.data.qpos[:] = [np.pi, 0.0]
        self.data.qvel[:] = [0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0

        # self.current_episode = options.get("current_episode", 0) if options else 0
        # self.total_episodes = options.get("total_episodes", 500) if options else 1000

        return self._get_obs(), {}


    def step(self, action):
        self.step_count += 1
        self.data.ctrl[0] = np.clip(action[0], -5.0, 5.0)
        # np.random.seed(self.seed)


        wind_force = np.array([np.random.normal(0.0, 1.0), 0, 0, 0, 0, 0])
        self.data.xfrc_applied[self.link2_id] = wind_force

        mujoco.mj_step(self.model, self.data)

        if self.render_mode and self.viewer:
            self.viewer.sync()

        obs = self._get_obs()
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        tip_pos = self.data.site_xpos[site_id]
        tip_height = tip_pos[2]

        # # Track training progress
        # progress = self.current_episode / self.total_episodes
        # target_height = 2.0 + 2.0 * progress

        # # Shaped reward
        # reward = tip_height / target_height  # encourages reaching current goal

        # # Bonus if achieved
        # if tip_height >= target_height:
        #     reward += 10.0 * progress  # more bonus later
        target_pos = np.array([0, 0, 4])
        reward = -np.linalg.norm(tip_pos - target_pos)
        distance = np.linalg.norm(tip_pos - target_pos)
        reward = -distance
        if distance < 0.2:
            reward += 5.0  # small bonus near goal

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
