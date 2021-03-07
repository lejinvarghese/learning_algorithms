from itertools import chain
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
from beeline_world import main
import matplotlib.pyplot as plt
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node
from pomegranate.utils import plot_networkx
from itertools import product
from scipy.spatial.distance import cdist
from warnings import filterwarnings
filterwarnings("ignore")


p01 = DiscreteDistribution({1: 0.2, 0: 0.8})
p10 = DiscreteDistribution({1: 0.2, 0: 0.8})
p11 = DiscreteDistribution({1: 0.2, 0: 0.8})

grid_width, grid_height = 3, 3
pits = {}
nodes = {}
breezes = {}


_positions = [(0, 0)]


def get_neighborhood_percepts(position, grid_width, grid_height, pit_proba=0.2):

    def _get_neighbors(position, grid_width, grid_height):
        _x, _y = position
        _neighbors = list(chain.from_iterable([[str(i)+"_"+str(j) for j in range(_y-1, _y+2) if (i >= 0) and (i < grid_width) and (j >= 0) and (
            j < grid_height) and (cdist(np.array(position).reshape(1, -1), np.array((i, j)).reshape(1, -1), metric="cityblock") == 1)] for i in range(_x-1, _x+2)]))
        return _neighbors

    _neighbors = _get_neighbors(position, grid_width, grid_height)

    def _get_cpt(n_neighbors):
        n_neighbors += 1
        _matrix = np.zeros(n_neighbors)

        for i in range(1, 2**n_neighbors):
            _binary = np.array([int(x)
                                for x in list(np.binary_repr(i, width=n_neighbors))])
            _matrix = np.vstack((_matrix, _binary))

        _any_pit = np.max(_matrix[:, :n_neighbors-1], axis=1)
        _any_breeze = _matrix[:, -1]

        _proba = np.expand_dims(np.invert(np.logical_xor(
            _any_pit, _any_breeze)).astype(float), axis=1)
        _matrix = np.hstack((_matrix, _proba))
        return _matrix

    def _get_pits_nodes_breezes(position, neighbors):
        _pits, _nodes, _breezes = {}, {}, {}
        for cell in neighbors:
            _pits[cell] = DiscreteDistribution(
                {1: pit_proba, 0: 1 - pit_proba})
            _nodes[cell] = Node(_pits[cell], name=cell)

        _x, _y = position
        _breezes[str(_x)+'_'+str(_y)] = ConditionalProbabilityTable(
            _get_cpt(n_neighbors=len(neighbors)), [_pits[cell] for cell in neighbors])
        return _pits, _nodes, _breezes

    return _get_pits_nodes_breezes(position, _neighbors)


print(get_neighborhood_percepts((0, 0), grid_width, grid_height, 0.2))


# model.predict_proba([{'guest': 'A', 'monty': 'C'}])
for x, y in product(range(grid_width), range(grid_height)):
    pits[str(x)+'_'+str(y)] = DiscreteDistribution({1: 0.2, 0: 0.8})
    nodes[str(x)+'_'+str(y)] = Node(pits[str(x) +
                                         '_'+str(y)], name=str(x)+'_'+str(y))


for x, y in product(range(grid_width), range(grid_height)):
    try:
        breezes[str(x)+'_'+str(y)] = ConditionalProbabilityTable(generate_cpt_n(2),
                                                                 [pits[str(x+1)+'_'+str(y)], pits[str(x)+'_'+str(y+1)]])
    except:
        pass

# print(breezes.keys(), len(breezes))


# print(generate_cpt_n(n_sides=2))

# b00 = ConditionalProbabilityTable([[0, 0, 0, 1.0],
#                                    [0, 0, 1, 0.0],
#                                    [0, 1, 0, 0.0],
#                                    [0, 1, 1, 1.0],
#                                    [1, 0, 0, 0.0],
#                                    [1, 0, 1, 1.0],
#                                    [1, 1, 0, 0.0],
#                                    [1, 1, 1, 1.0]], [p01, p10])

# s1 = Node(p01, name="p01")
# s2 = Node(p10, name="p10")
# s3 = Node(b00, name="b00")

# model = BayesianNetwork("Monty Hall Problem")
# model.add_states(s1, s2, s3)
# model.add_edge(s1, s3)
# model.add_edge(s2, s3)
# model.bake()

# print(model.predict_proba([{'b00': 1}]))

# n_simulations = 300
# n_success = 0
# for i in range(n_simulations):
#     try:
#         main()
#         n_success += 1
#     except:
#         pass
# print('total successes', n_success)

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