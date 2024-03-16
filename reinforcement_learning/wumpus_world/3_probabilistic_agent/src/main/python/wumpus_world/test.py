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
filterwarnings("ignore")

n_simulations = 30
n_success = 0
rewards = []
for i in range(n_simulations):
    try:
        rewards.append(main())
    except:
        pass
print(f'mean reward: ', {np.mean(rewards)},
      'median reward: ', {np.median(rewards)})

# p01 = BernoulliDistribution(0.2)
# arr = np.array([[1, 0, 1], [0, 1, 0]])
# x = np.transpose(np.count_nonzero(
#     arr == 1, axis=1))
# x[x > 1] = 0
# print(x)
# print(np.max(arr, axis=1))

# wl = DiscreteDistribution({'0_1': 0.2, '1_0': .8})

# w_cpt = ConditionalProbabilityTable(
#     [['0_1',  1, 1.0], ['0_1',  0, 0.0],
#      ['1_0',  1, 0.0], ['1_0',  0, 1.0], ], [wl])

# s1 = State(wl, name="wl")
# s3 = State(w_cpt, name="w_cpt")

# model = BayesianNetwork("Monty Hall Problem")
# model.add_states(s1, s3)
# model.add_edge(s1, s3)
# model.bake()
# # print(model.predict([['0_1',  None]]))
# x = model.predict_proba({'0_1': 1})
# for _ in x:
#     print(x[0])
# grid_width, grid_height = 4, 4
# pits = {}
# nodes = {}
# breezes = {}

# print(set([Coordinates(0, 0), Coordinates(0, 0), Coordinates(0, 0)]))


# def get_neighborhood_percepts(position, grid_width, grid_height, visited_locations, inferred_pit_probs, breeze=False, pit_proba=0.2):

#     def _get_neighbors(position, grid_width, grid_height):
#         _x, _y = position.x, position.y
#         _neighbors = list(chain.from_iterable([[str(i)+"_"+str(j) for j in range(_y-1, _y+2) if (i >= 0) and (i < grid_width) and (j >= 0) and (
#             j < grid_height) and (cdist(np.array((_x, _y)).reshape(1, -1), np.array((i, j)).reshape(1, -1), metric="cityblock") == 1)] for i in range(_x-1, _x+2)]))
#         return _neighbors

#     _neighbors = _get_neighbors(position, grid_width, grid_height)

#     def _get_cpt(n_neighbors):
#         n_neighbors += 1
#         _matrix = np.zeros(n_neighbors)

#         for i in range(1, 2**n_neighbors):
#             _binary = np.array([int(x)
#                                 for x in list(np.binary_repr(i, width=n_neighbors))])
#             _matrix = np.vstack((_matrix, _binary))

#         _any_pit = np.max(_matrix[:, :n_neighbors-1], axis=1)
#         _any_breeze = _matrix[:, -1]

#         _proba = np.expand_dims(np.invert(np.logical_xor(
#             _any_pit, _any_breeze)).astype(float), axis=1)
#         _matrix = np.hstack((_matrix, _proba))
#         return _matrix

#     def _get_pits_nodes_breezes(position, neighbors):
#         _pits, _nodes, _breezes = {}, {}, {}
#         for n_loc in neighbors:
#             _pits[n_loc] = DiscreteDistribution(
#                 {1: pit_proba, 0: 1 - pit_proba})
#             _nodes[n_loc] = State(_pits[n_loc], name=n_loc)

#         _x, _y = position.x, position.y
#         p_loc = str(_x)+'_'+str(_y)
#         _breezes[p_loc] = ConditionalProbabilityTable(
#             _get_cpt(n_neighbors=len(neighbors)), [_pits[n_loc] for n_loc in neighbors])
#         _nodes[p_loc] = State(_breezes[p_loc], name=p_loc)
#         return _pits, _nodes, _breezes

#     _pits, _nodes, _breezes = _get_pits_nodes_breezes(position, _neighbors)

#     def _get_model(pits, nodes, breezes):
#         model = BayesianNetwork("pits and breezes")
#         for state in nodes:
#             model.add_states(nodes[state])
#         for pit in list(pits.keys()):
#             for breeze in list(breezes.keys()):
#                 model.add_edge(nodes[pit], nodes[breeze])
#         model.bake()
#         return model

#     model = _get_model(_pits, _nodes, _breezes)

#     def _get_safe_locations(visited_locations, inferred_pit_probs, tolerance=0.5*pit_proba):
#         _visited_locations = set([str(loc.x) + '_' + str(loc.y)
#                                   for loc in visited_locations])
#         _inferred_pit_probs = set(
#             [key for key, value in inferred_pit_probs.items() if value < tolerance])
#         _safe_locations = _inferred_pit_probs.union(_visited_locations)
#         print(_safe_locations, _visited_locations, _inferred_pit_probs)
#         return _safe_locations

#     def _get_pit_post_proba(position, neighbors, model, safe_locations):

#         _position = str(position.x) + '_' + str(position.y)
#         _state_names = [state.name for state in model.states]
#         _input_neighborhood = {
#             item: 0 for item in _state_names if (item in list(_safe_locations)) & (item not in [_position])}

#         if breeze:
#             _input_neighborhood[_position] = 1
#         else:
#             _input_neighborhood[_position] = 0

#         _proba = model.predict_proba([_input_neighborhood])[0]
#         print(_proba)

#         _proba = [{neighbors[i]: round(x.parameters[0].get(1), 2)}
#                   for i, x in enumerate(_proba) if isinstance(x, DiscreteDistribution)]
#         _proba = dict((key, val) for k in _proba for key, val in k.items())
#         return _proba

#     _safe_locations = _get_safe_locations(
#         visited_locations, inferred_pit_probs)
#     _inferred_pit_probs = _get_pit_post_proba(
#         position, _neighbors, model, _safe_locations)
#     print(_inferred_pit_probs)
#     return _inferred_pit_probs


# current_location = Coordinates(1, 1)
# visited_locations = [Coordinates(0, 0), Coordinates(1, 0)]
# inferred_pit_probs = {'0_0': 0, '1_1': 0.56, '2_0': 0.56}  # , '3_0': 0.55}

# inferred_pit_probs = get_neighborhood_percepts(current_location, grid_width,
#                                                grid_height, visited_locations, inferred_pit_probs, breeze=False)
# .get('parameters').get("1"))  # [-1].parameters[0][1])
# predictions = list(model.predict_proba([breeze_locations])[0])


# print(list(filter(lambda state: state, predictions)))
# get("parameters", None))  # .get("1", None))
# print([x for x in predictions][0].parameters[0][1])
# x = []
# for i, s in enumerate(state_names):
#     try:
#         print(s, predictions[i].parameters[0][1])
#     except:
#         pass
# for x, y in product(range(grid_width), range(grid_height)):
#     pits[str(x)+'_'+str(y)] = DiscreteDistribution({1: 0.2, 0: 0.8})
#     nodes[str(x)+'_'+str(y)] = Node(pits[str(x) +
#                                          '_'+str(y)], name=str(x)+'_'+str(y))


# for x, y in product(range(grid_width), range(grid_height)):
#     try:
#         bree
# print(list(filter(lambda state: state, predictions)))
# get("parameters", None))  # .get("1", None))
# print([x for x in predictions][0].parameters[0][1])
# x = []
# for i, s in enumerate(state_names):
#     try:
#         print(s, predictions[i].parameters[0][1])
#     except:
#         pass
# for x, y in product(range(grid_width), range(grid_height)):
#     pits[str(x)+'_'+str(y)] = DiscreteDistribution({1: 0.2, 0: 0.8})
#     nodes[str(x)+'_'+str(y)] = Node(pits[str(x) +
#                                          '_'+str(y)], name=str(x)+'_'+str(y))


# for x, y in product(range(grid_width), range(grid_height)):
#     try:
#         breezes[str(x)+'_'+str(y)] = ConditionalProbabilityTable(generate_cpt_n(2),
#                                                                  [pits[str(x+1)+'_'+str(y)], pits[str(x)+'_'+str(y+1)]])
#     except:
#         pass

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
