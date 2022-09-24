import gym

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="human")
    env.reset()

    for i in range(10):
        env.render()
        env.step(env.action_space.sample())
        env.reset()
    env.close()
