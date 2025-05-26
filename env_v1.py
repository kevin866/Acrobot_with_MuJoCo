import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleAcrobotEnv(gym.Env):
    def __init__(self, render_mode=False):
        super().__init__()

        self.max_steps = 2000
        self.step_count = 0
        self.render_mode = render_mode
        self.target_height = 1.0  # meters, estimated tip height goal

        # Physical constants
        self.dt = 0.05  # time step
        self.g = 9.81
        self.link_lengths = [1.0, 1.0]  # meters
        self.link_masses = [1.0, 1.0]
        self.inertia = [1.0, 1.0]  # simplified
        self.damping = 0.01

        # Action and observation spaces
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),  # [theta1, theta2, theta1_dot, theta2_dot]
            dtype=np.float32,
        )

        self.state = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([np.pi, 0.0, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        theta1, theta2, dtheta1, dtheta2 = self.state
        torque = np.clip(action[0], -5.0, 5.0)

        # Simple dynamics update (not accurate — for illustrative purposes)
        ddtheta1 = -self.damping * dtheta1 + torque
        ddtheta2 = -self.damping * dtheta2

        dtheta1 += ddtheta1 * self.dt
        dtheta2 += ddtheta2 * self.dt
        theta1 += dtheta1 * self.dt
        theta2 += dtheta2 * self.dt

        self.state = np.array([theta1, theta2, dtheta1, dtheta2], dtype=np.float32)

        # Compute tip height
        tip_x = self.link_lengths[0] * np.cos(theta1) + self.link_lengths[1] * np.cos(theta1 + theta2)
        tip_y = self.link_lengths[0] * np.sin(theta1) + self.link_lengths[1] * np.sin(theta1 + theta2)
        tip_height = tip_y  # y = height

        position_reward = -np.linalg.norm([tip_x, tip_y - self.target_height])
        control_penalty = 0.01 * torque**2
        reward = position_reward - control_penalty

        success = tip_height >= self.target_height
        terminated = success
        truncated = self.step_count >= self.max_steps

        self.step_count += 1
        return self.state.copy(), reward, bool(terminated), bool(truncated), {}

    def render(self):
        # Optional: visualize with matplotlib or print simple text output
        theta1, theta2, _, _ = self.state
        print(f"Angles: θ1={theta1:.2f}, θ2={theta2:.2f}")

    def close(self):
        pass
