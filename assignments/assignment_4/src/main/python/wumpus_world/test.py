from itertools import chain
import pandas as pd
from environment.environment import (
    Action,
    Percept,
    Orientation,
    Coordinates,
    Environment,
    AgentState,
)
import numpy as np
from agent.agent import Agent, BeelineAgent
from probabilistic_world import main
import matplotlib.pyplot as plt
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node, State, BernoulliDistribution, UniformDistribution
from pomegranate.utils import plot_networkx
from itertools import product
from scipy.spatial.distance import cdist
from warnings import filterwarnings
import torch
filterwarnings("ignore")

action_set = {
    0: Action(1),
    1: Action(2),
    2: Action(3),
    3: Action(4),
    4: Action(5),
    5: Action(6),
}

sample = np.random.rand(1, 72)
# sample = np.ones((1, 72))
sample = torch.from_numpy(sample).float()
model = torch.load("./models")
model.eval()
y_pred = model(sample).data.numpy()
action = np.argmax(y_pred)
action = action_set[action]
print(action)
