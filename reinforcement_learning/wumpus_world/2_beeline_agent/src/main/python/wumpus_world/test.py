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
import networkx as nx
from scipy.spatial.distance import cdist
from itertools import product
from beeline_world import main

n_simulations = 300
n_success = 0
for i in range(n_simulations):
    try:
        main()
        n_success += 1
    except:
        pass
print('total successes', n_success)

# # test basic
# action = Action(randint(1, 6))
# print(action)
# print(Direction(2), Direction(1))
# percept = Percept().show()
# print(percept)

# #  test orientation
# orient = Orientation()
# print(orient.orientation.name)
# orient.turn_left()
# print(orient.orientation.name)
# orient.turn_left()
# print(orient.orientation.name)

# test agent
# agent = Agent()
# print(agent.orientation.orientation)
# print(agent.location.x, agent.location.y)

# agent.orientation.turn_left()
# print(agent.orientation.orientation)
# print(agent.location.x, agent.location.y)

# for i in range(6):
#     agent = agent.forward(4, 4)
#     if i == 3:
#         agent.orientation.turn_left()
#     print(agent.orientation.orientation)
#     print(agent.location.x, agent.location.y)

# # test Environment
# env = Environment(grid_width=5, grid_height=5, pit_proba=0.2, allow_climb_without_gold=True, agent=Agent(),
#                   pit_locations=None, terminated=False, wumpus_location=None, wumpus_alive=True, gold_location=None)
# env, per = env.initialize()
# print(env.wumpus_location.x, env.wumpus_location.y)

# print(env.visualize())
