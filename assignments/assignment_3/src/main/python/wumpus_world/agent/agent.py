#!/usr/bin/env python3
from environment.environment import AgentState, Action, Coordinates, Direction
import numpy as nppit
import networkx as nx
from scipy.spatial.distance import cdist
from itertools import product


class Agent:
    def next_action(self, percept):
        return Action()


class NaiveAgent:
    def next_action(self, percept):
        _rand_action = np.random.randint(low=1, high=6)
        return Action(_rand_action)


class BeelineAgent:
    def __init__(
        self, grid_width, grid_height, agent_state, safe_locations, beeline_action_list
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.agent_state = agent_state
        self.safe_locations = safe_locations
        self.beeline_action_list = beeline_action_list

    def __copy__(self):
        return BeelineAgent(
            self.grid_width,
            self.grid_height,
            self.agent_state,
            self.safe_locations,
            self.beeline_action_list,
        )

    def _show(self):
        print(
            f"beeline agent state: {self.agent_state.show()} \n safe locations: {self.safe_locations} \n beeline action list: {self.beeline_action_list}"
        )

    def _construct_beeline_plan(self):

        def _construct_beeline_path(self):
            _safe_locations = list(self.safe_locations)
            _safe_locations.extend([self.agent_state.location])
            G = nx.Graph()
            for node in _safe_locations:
                G.add_node((node.x, node.y))
            for x, y in list(product(G.nodes, G.nodes)):
                if (
                    cdist(
                        np.array(x).reshape(1, -1),
                        np.array(y).reshape(1, -1),
                        metric="cityblock",
                    )
                    == 1
                ):
                    G.add_edges_from([(x, y)])
            print(
                f"no. of safe locations: {G.number_of_nodes()}")
            return nx.shortest_path(
                G,
                source=(self.agent_state.location.x,
                        self.agent_state.location.y),
                target=(0, 0),
            )

        _beeline_path = _construct_beeline_path(self)
        print(f"beeline path >> {_beeline_path}")

        def _construct_plan_from_path(self, beeline_path):
            active_orientation = self.agent_state.orientation.orientation
            beeline_actions = []
            for i, node in enumerate(beeline_path):
                if i < len(beeline_path) - 1:
                    cur_pos_x,  cur_pos_y = node
                    nxt_pos_x,  nxt_pos_y = beeline_path[i+1]
                    if cur_pos_x < nxt_pos_x:  # go east
                        if active_orientation == Direction.east:
                            actions = [Action.forward]
                        elif active_orientation == Direction.north:
                            actions = [Action.turn_right, Action.forward]
                        elif active_orientation == Direction.west:
                            actions = [Action.turn_right,
                                       Action.turn_right, Action.forward]
                        elif active_orientation == Direction.south:
                            actions = [Action.turn_left, Action.forward]
                        active_orientation = Direction.east
                    elif cur_pos_x > nxt_pos_x:  # go west
                        if active_orientation == Direction.west:
                            actions = [Action.forward]
                        elif active_orientation == Direction.north:
                            actions = [Action.turn_left, Action.forward]
                        elif active_orientation == Direction.east:
                            actions = [Action.turn_left,
                                       Action.turn_left, Action.forward]
                        elif active_orientation == Direction.south:
                            actions = [Action.turn_right, Action.forward]
                        active_orientation = Direction.west
                    elif cur_pos_y < nxt_pos_y:  # go north
                        if active_orientation == Direction.north:
                            actions = [Action.forward]
                        elif active_orientation == Direction.west:
                            actions = [Action.turn_right, Action.forward]
                        elif active_orientation == Direction.south:
                            actions = [Action.turn_right,
                                       Action.turn_right, Action.forward]
                        elif active_orientation == Direction.east:
                            actions = [Action.turn_left, Action.forward]
                        active_orientation = Direction.north
                    elif cur_pos_y > nxt_pos_y:  # go south
                        if active_orientation == Direction.south:
                            actions = [Action.forward]
                        elif active_orientation == Direction.west:
                            actions = [Action.turn_left, Action.forward]
                        elif active_orientation == Direction.north:
                            actions = [Action.turn_left,
                                       Action.turn_left, Action.forward]
                        elif active_orientation == Direction.east:
                            actions = [Action.turn_right, Action.forward]
                        active_orientation = Direction.south
                    for action in actions:
                        beeline_actions.extend([action])
            return beeline_actions
        _beeline_actions = _construct_plan_from_path(self, _beeline_path)
        print(f"beeline actions >> {_beeline_actions}")
        return _beeline_actions

    def next_action(self, percept):
        if self.agent_state.has_gold:
            if (self.agent_state.location.x == 0) & (self.agent_state.location.y == 0):
                new_agent = self
                _action = Action.climb
            else:
                _beeline_plan = (
                    self._construct_beeline_plan()
                    if not self.beeline_action_list
                    else self.beeline_action_list
                )
                _action = _beeline_plan[0]
                new_agent = BeelineAgent.__copy__(self)
                new_agent.agent_state = new_agent.agent_state.apply_move_action(
                    _action, self.grid_width, self.grid_height
                )
                new_agent.beeline_action_list = _beeline_plan[1:]

        elif percept.glitter:
            new_agent = BeelineAgent.__copy__(self)
            new_agent.agent_state.has_gold = True
            _action = Action.grab
        else:
            _rand_number = np.random.randint(low=1, high=5)
            if _rand_number == 1:
                new_agent = BeelineAgent.__copy__(self)
                _new_agent_state = new_agent.agent_state.forward(
                    self.grid_width, self.grid_height
                )
                _new_safe_locations_l = list(self.safe_locations)
                _new_safe_locations_l.extend([_new_agent_state.location])
                new_agent.agent_state = _new_agent_state
                new_agent.safe_locations = set(_new_safe_locations_l)
                _action = Action.forward
            elif _rand_number == 2:
                new_agent = BeelineAgent.__copy__(self)
                new_agent.agent_state = new_agent.agent_state.turn_left()
                _action = Action.turn_left
            elif _rand_number == 3:
                _new_agent_state = self.agent_state.turn_right()
                new_agent = BeelineAgent.__copy__(self)
                new_agent.agent_state = _new_agent_state
                _action = Action.turn_right
            elif _rand_number == 4:
                new_agent = BeelineAgent.__copy__(self)
                new_agent.agent_state.use_arrow()
                _action = Action.shoot

        return new_agent, _action


def initialize_beeline_agent(grid_width, grid_height):
    return BeelineAgent(grid_width, grid_height, AgentState(), set([Coordinates(0, 0)]), [])
