from typing import List

import numpy as np
from gym import Env


def extract_policy(
    env: Env, value_states: List[float], gamma: float = 1.0
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
