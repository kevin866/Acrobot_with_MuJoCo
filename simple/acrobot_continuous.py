import gym
from gym import spaces
# from gym.envs.classic_control import rendering
import numpy as np

class AcrobotContinuousEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self):
        self.LINK_LENGTH_1 = 1.0
        self.LINK_LENGTH_2 = 1.0
        self.LINK_MASS_1 = 1.0
        self.LINK_MASS_2 = 1.0
        self.LINK_COM_POS_1 = 0.5
        self.LINK_COM_POS_2 = 0.5
        self.LINK_MOI = 1.0
        self.MAX_VEL_1 = 4 * np.pi
        self.MAX_VEL_2 = 9 * np.pi
        self.dt = 0.2
        self.g = 9.8
        self.max_torque = 1.0

        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.state = None
        self.viewer = None

    def reset(self):
        high = np.array([np.pi, np.pi, 1.0, 1.0])
        self.state = np.array([
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
        ])
        return self._get_ob()

    def _get_ob(self):
        theta1, theta2, dtheta1, dtheta2 = self.state
        return np.array([np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2), dtheta1, dtheta2])

    def step(self, u):
        torque = np.clip(u[0], -self.max_torque, self.max_torque)

        s = self.state
        theta1, theta2, dtheta1, dtheta2 = s

        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        l2 = self.LINK_LENGTH_2
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = self.g
        dt = self.dt

        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2)
        phi1 = -m2 * l1 * lc2 * dtheta2**2 * np.sin(theta2) - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2) + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2

        ddtheta2 = (torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * np.sin(theta2) - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        dtheta1 += dt * ddtheta1
        dtheta2 += dt * ddtheta2
        dtheta1 = np.clip(dtheta1, -self.MAX_VEL_1, self.MAX_VEL_1)
        dtheta2 = np.clip(dtheta2, -self.MAX_VEL_2, self.MAX_VEL_2)
        theta1 += dt * dtheta1
        theta2 += dt * dtheta2

        self.state = np.array([theta1, theta2, dtheta1, dtheta2])

        terminal = self._terminal()
        reward = 0.0 if not terminal else 1.0

        return self._get_ob(), reward, terminal, {}

    def _terminal(self):
        theta1, theta2, _, _ = self.state
        return -np.cos(theta1) - np.cos(theta1 + theta2) > 1.0

    def render(self, mode="human"):
        pass  # Can be added if needed

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
