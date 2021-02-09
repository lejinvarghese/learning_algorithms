#!/usr/bin/env python3
from environment.environment import Action
import numpy as np


# class Agent:

#     def next_action(self, percept):
#         return Action()


class NaiveAgent:

    def __init__(self):
        self._rand_action = np.random.randint(low=1, high=6)
        print(self._rand_action)

    def next_action(self, percept):
        return Action(self._rand_action)
