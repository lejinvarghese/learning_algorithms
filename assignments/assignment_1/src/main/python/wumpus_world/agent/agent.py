#!/usr/bin/env python3
from environment.environment import Action
import numpy as np


class Agent:

    def next_action(self, percept):
        return Action()


class NaiveAgent:

    def next_action(self, percept):
        _rand_action = np.random.randint(low=1, high=6)
        return Action(_rand_action)
