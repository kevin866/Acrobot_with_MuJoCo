
# Import your custom environment
from acrobot_env import AcrobotMujocoEnv  # Update to your actual file name

env = AcrobotMujocoEnv()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
policy = PolicyNetwork(obs_dim, act_dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# Train and log to TensorBoard
train_reinforce(env, policy, optimizer, num_episodes=500)

# Save trained model
save_policy(policy)

# Evaluate
evaluate_policy(env, policy, num_episodes=10)

# Later, or in a new script...
# load_policy(policy)
# evaluate_policy(env, policy)
