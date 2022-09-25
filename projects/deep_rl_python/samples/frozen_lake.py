import gym

n_episodes = 20
n_log_frequency = n_episodes // 5
n_timesteps = 50


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="human")

    for e in range(n_episodes):
        episode_return = 0
        state = env.reset()

        for t in range(n_timesteps):
            env.render()
            random_action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(
                random_action
            )
            episode_return += reward
            if terminated:
                print(f"Episode finished after {t+1} timesteps")
                print(f"Episode return: {episode_return}")
                break
        if e % n_log_frequency == 0:
            print(f"Episode: {e}, Return: {episode_return}")

    env.close()
