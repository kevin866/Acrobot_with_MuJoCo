def evaluate_policy(env, policy, num_episodes=10):
    policy.eval()
    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                mean, std = policy(obs)
                action = mean  # Use deterministic policy (mean)
            obs_new, reward, terminated, truncated, _ = env.step(action.numpy())
            done = terminated or truncated
            episode_reward += reward
            obs = torch.tensor(obs_new, dtype=torch.float32)

        print(f"Eval Episode {episode}, Total Reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average Eval Reward over {num_episodes} episodes: {avg_reward:.2f}")
