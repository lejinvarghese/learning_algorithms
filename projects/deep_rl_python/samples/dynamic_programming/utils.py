from random import gammavariate
from typing import List
from enum import Enum
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


class IterationStrategy(Enum):
    VALUE = 1
    POLICY = 2


class ValueFunction:
    def __init__(
        self,
        env: Env,
        iteration_strategy: IterationStrategy,
        policy: List[int] = None,
        n_iterations: int = 1000,
        convergence_threshold: float = 1e-20,
        gamma: float = 1.0,
    ):
        self.env = env
        self.iteration_strategy = iteration_strategy
        if (policy is None) and (
            iteration_strategy == IterationStrategy.POLICY
        ):
            raise ValueError("Policy is required for policy iteration")
        else:
            self.policy = policy
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.gamma = gamma
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.value_states = np.zeros(self.n_states)

    def compute(self):
        for i in range(self.n_iterations):
            next_value_states = np.copy(self.value_states)
            for s in range(self.n_states):
                if self.iteration_strategy == IterationStrategy.VALUE:
                    q_states_actions = [
                        sum(self.__compute_value_state(s, a, next_value_states))
                        for a in range(self.n_actions)
                    ]
                    self.value_states[s] = max(q_states_actions)
                elif self.iteration_strategy == IterationStrategy.POLICY:
                    a = self.policy[s]
                    self.value_states[s] = sum(
                        self.__compute_value_state(s, a, next_value_states)
                    )
                else:
                    raise ValueError("Invalid iteration strategy")

            if (
                np.sum(np.fabs(next_value_states - self.value_states))
                <= self.convergence_threshold
            ):
                print(f"Iteration converged at iteration {i+1}.")
                break
        return np.round(self.value_states, 4)

    def __compute_value_state(
        self, state: int, action: int, next_value_states: List[float]
    ) -> List[float]:
        return [
            trans_proba * (reward + self.gamma * next_value_states[next_state])
            for trans_proba, next_state, reward, _ in self.env.P[state][action]
        ]
