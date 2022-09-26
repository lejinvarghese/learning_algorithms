from typing import List
import numpy as np
import gym

n_episodes = 10
n_log_frequency = max(1, n_episodes // 10)
n_timesteps = 50


def compute_optimal_values(
    env: gym.Env,
    n_iterations: int = 1000,
    convergence_threshold: float = 1e-20,
    gamma: float = 1.0,
) -> List[float]:
    """
    Goal: Compute the optimal value function.

    Args:
        env : gym environment
        n_iterations : number of iterations
        convergence_threshold : threshold for convergence
        gamma : discount factor

    Returns:
        value_states: a list of values for each state
    """
    _n_states = env.observation_space.n
    _n_actions = env.action_space.n
    _value_states = np.zeros(_n_states)

    for i in range(n_iterations):
        _next_value_states = np.copy(_value_states)
        for s in range(_n_states):
            _q_states_actions = [
                sum(
                    [
                        trans_proba * (reward + gamma * _next_value_states[next_state])
                        for trans_proba, next_state, reward, _ in env.P[s][a]
                    ]
                )
                for a in range(_n_actions)
            ]
            _value_states[s] = np.round(max(_q_states_actions), 4)

        if np.sum(np.fabs(_next_value_states - _value_states)) <= convergence_threshold:
            print(f"Value-iteration converged at iteration {i+1}.")
            break

    return _value_states


def compute_optimal_policy(
    env: gym.Env, value_states: List[float], gamma: float = 1.0
) -> List[int]:
    """
    Goal: Compute the optimal policy.

    Args:
        env : gym environment
        value_states : a list of values for each state,
        gamma : discount factor

    Returns:
        policy: a list of actions for each state
    """
    _n_states = env.observation_space.n
    _n_actions = env.action_space.n
    _policy = np.zeros(_n_states)

    for s in range(_n_states):
        _q_states_actions = [
            sum(
                trans_proba * (reward + gamma * value_states[next_state])
                for trans_proba, next_state, reward, _ in env.P[s][a]
            )
            for a in range(_n_actions)
        ]
        _policy[s] = np.argmax(np.array(_q_states_actions))

    return _policy


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="human")
    state = env.reset()
    total_returns = 0
    optimal_value_states = compute_optimal_values(env)
    optimal_policy = compute_optimal_policy(env, optimal_value_states)
    print(f"Optimal policy obtained through value iteration: {optimal_policy}")

    for e in range(n_episodes):
        episode_return = 0
        state = env.reset()
        next_state = state[0]

        for t in range(n_timesteps):
            env.render()
            next_action = int(optimal_policy[next_state])
            next_state, reward, terminated, truncated, info = env.step(next_action)

            episode_return += reward
            if terminated:
                break
        total_returns += episode_return

        if (e + 1) % n_log_frequency == 0:
            print(
                f"Episode {e+1}, finished after {t+1} timesteps, with a return {episode_return}."
            )

        print(f"Agent obtained a return {episode_return}.")

    print(f"Agent obtained a total return {total_returns} in {n_episodes} episodes.")

    env.close()
