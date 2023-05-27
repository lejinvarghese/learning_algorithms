from random import gammavariate
from typing import List, Dict
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
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    next_policy = np.zeros(n_states)

    for s in range(n_states):
        q_states_actions = [
            sum(
                trans_proba * (reward + gamma * value_states[next_state])
                for trans_proba, next_state, reward, _ in env.P[s][a]
            )
            for a in range(n_actions)
        ]
        next_policy[s] = np.argmax(np.array(q_states_actions))

    return next_policy


def compute_value_state_action(
    states_actions: Dict[int : Dict[int : List[float, int, float, bool]]],
    next_value_states: List[float],
    state: int,
    action: int,
    gamma: float = 1.0,
) -> List[float]:
    return [
        trans_proba * (reward + gamma * next_value_states[next_state])
        for trans_proba, next_state, reward, _ in states_actions[state][action]
    ]


class IterationStrategy(Enum):
    VALUE = 1
    POLICY = 2


class MarkovDecisionProcess:
    def __init__(
        self,
        env: Env,
        n_iterations: int = 1000,
        convergence_threshold: float = 1e-20,
        gamma: float = 1.0,
    ):
        self.env = env
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.gamma = gamma
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.states_actions = self.env.P
        self.value_states = np.zeros(self.n_states)

    def compute_value_function(
        self,
        iteration_strategy: IterationStrategy,
        policy: List[int] = None,
    ):
        if (iteration_strategy == IterationStrategy.POLICY) and (
            policy is None
        ):
            raise ValueError(
                "Policy is required for policy iteration strategy."
            )
        for i in range(self.n_iterations):
            next_value_states = np.copy(self.value_states)
            for s in range(self.n_states):
                if self.iteration_strategy == IterationStrategy.VALUE:
                    q_states_actions = [
                        sum(
                            compute_value_state_action(
                                self.states_actions,
                                next_value_states,
                                s,
                                a,
                            )
                        )
                        for a in range(self.n_actions)
                    ]
                    self.value_states[s] = max(q_states_actions)
                elif self.iteration_strategy == IterationStrategy.POLICY:
                    a = self.policy[s]
                    self.value_states[s] = sum(
                        compute_value_state_action(
                            self.states_actions, next_value_states, s, a
                        )
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


class OptimalPolicy:
    def __init__(
        self,
        mdp: MarkovDecisionProcess,
        iteration_strategy: IterationStrategy,
    ):
        self.mdp = mdp
        self.iteration_strategy = iteration_strategy

    def extract(self):
        pass
