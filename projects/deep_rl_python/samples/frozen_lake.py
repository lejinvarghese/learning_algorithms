import gym

n_episodes = 20
n_log_frequency = n_episodes // 10
n_timesteps = 50


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="human")

    for e in range(1, n_episodes + 1):
        episode_return = 0
        state = env.reset()

        for t in range(1, n_timesteps + 1):
            env.render()
            random_action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(
                random_action
            )
            episode_return += reward
            if terminated:
                break
        if e % n_log_frequency == 0:
            print(
                f"Episode {e}, finished after {t} timesteps, with a return {episode_return}."
            )

    env.close()
