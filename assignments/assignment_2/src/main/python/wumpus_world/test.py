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

safe_locations = list(
    set(
        [
            Coordinates(0, 0),
            Coordinates(1, 0),
            Coordinates(4, 1),
            Coordinates(5, 6),
            Coordinates(3, 1),
            Coordinates(2, 1),
            Coordinates(1, 1),
        ]
    )
)
G = nx.Graph()
A = np.zeros(shape=(5, 5))
for node in safe_locations:
    G.add_node((node.x, node.y))
for x, y in list(product(G.nodes, G.nodes)):
    if (
        cdist(
            np.array(x).reshape(1, -1), np.array(y).reshape(1, -1), metric="cityblock"
        )
        == 1
    ):
        print("nodes: ", x, y)
        G.add_edges_from([(x, y)])
print(G.number_of_nodes(), G.nodes)
print(G.number_of_edges(), G.edges)
_beeline_path = nx.shortest_path(G, source=(4, 1), target=(0, 0))
print(_beeline_path)

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
