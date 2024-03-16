from typing import List
import numpy as np
import gym

try:
    from utils import extract_policy
except ImportError:
    from projects.deep_rl_python.samples.dynamic_programming.utils import (
        extract_policy,
    )


n_episodes = 10
n_log_frequency = max(1, n_episodes // 10)
n_timesteps = 50


def compute_value_function(
    env: gym.Env,
    policy: List[int],
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
    _value_states = np.zeros(_n_states)

    for i in range(n_iterations):
        _next_value_states = np.copy(_value_states)
        for s in range(_n_states):
            a = policy[s]
            _value_states[s] = sum(
                [
                    trans_proba * (reward + gamma * _next_value_states[next_state])
                    for trans_proba, next_state, reward, _ in env.P[s][a]
                ]
            )

        if np.sum(np.fabs(_next_value_states - _value_states)) <= convergence_threshold:
            break

    return np.round(_value_states, 4)


def policy_iteration(
    env: gym.Env,
    n_iterations: int = 1000,
) -> List[float]:
    """
    Goal: Compute the optimal policy.

    Args:
        env : gym environment
        n_iterations : number of iterations
        convergence_threshold : threshold for convergence
        gamma : discount factor

    Returns:
        value_states: a list of values for each state
    """

    _policy = np.zeros(env.observation_space.n)

    for i in range(n_iterations):

        _value_states = compute_value_function(env, _policy)

        _next_policy = extract_policy(env, _value_states)
        if np.all(_policy == _next_policy):
            break

        _policy = _next_policy

    print(f"Policy-iteration converged at iteration {i+1}.")
    return _policy


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="human")
    state = env.reset()
    total_returns = 0

    optimal_policy = policy_iteration(env)
    print(f"Optimal policy obtained through policy iteration: {optimal_policy}")

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
