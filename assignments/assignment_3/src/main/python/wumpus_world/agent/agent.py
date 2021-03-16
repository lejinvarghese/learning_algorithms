#!/usr/bin/env python3
from environment.environment import AgentState, Action, Coordinates, Direction
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from itertools import product, chain
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node, State
import random


class Agent:
    def next_action(self, percept):
        return Action()


class NaiveAgent:
    def next_action(self, percept):
        _rand_action = np.random.randint(low=1, high=6)
        return Action(_rand_action)


class BeelineAgent:
    def __init__(
        self, grid_width, grid_height, agent_state, visited_locations, beeline_action_list
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.agent_state = agent_state
        self.visited_locations = visited_locations
        self.beeline_action_list = beeline_action_list

    def __copy__(self):
        return BeelineAgent(
            self.grid_width,
            self.grid_height,
            self.agent_state,
            self.visited_locations,
            self.beeline_action_list,
        )

    def _show(self):
        print(
            f"agent state: {self.agent_state.show()} \n safe locations: {self.visited_locations} \n action list: {self.beeline_action_list}"
        )

    def _construct_beeline_plan(self):

        def _construct_beeline_path(self):
            _visited_locations = list(self.visited_locations)
            _visited_locations.extend([self.agent_state.location])
            G = nx.Graph()
            for node in _visited_locations:
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
                _new_visited_locations_l = list(self.visited_locations)
                _new_visited_locations_l.extend([_new_agent_state.location])
                new_agent.agent_state = _new_agent_state
                new_agent.visited_locations = set(_new_visited_locations_l)
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


class ProbabilisticAgent(BeelineAgent):
    def __init__(
            self, grid_width, grid_height, agent_state, visited_locations, beeline_action_list, breeze_locations, stench_locations, heard_scream, pit_proba, inferred_pit_probs, inferred_wumpus_probs):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.agent_state = agent_state
        self.visited_locations = visited_locations
        self.beeline_action_list = beeline_action_list
        self.breeze_locations = breeze_locations
        self.stench_locations = stench_locations
        self.heard_scream = heard_scream
        self.pit_proba = pit_proba
        self.inferred_pit_probs = inferred_pit_probs
        self.inferred_wumpus_probs = inferred_wumpus_probs

    def __copy__(self):
        return ProbabilisticAgent(
            self.grid_width,
            self.grid_height,
            self.agent_state,
            self.visited_locations,
            self.beeline_action_list,
            self.breeze_locations,
            self.stench_locations,
            self.heard_scream,
            self.pit_proba,
            self.inferred_pit_probs,
            self.inferred_wumpus_probs
        )

    def _get_neighbors(self):
        x, y = self.agent_state.location.x, self.agent_state.location.y
        neighbors = list(chain.from_iterable([[str(i)+"_"+str(j) for j in range(y-1, y+2) if (i >= 0) and (i < self.grid_height) and (j >= 0) and (
            j < self.grid_width) and (cdist(np.array((x, y)).reshape(1, -1), np.array((i, j)).reshape(1, -1), metric="cityblock") == 1)] for i in range(x-1, x+2)]))
        return neighbors

    def _get_safe_locations(self):
        tolerance = 0.10
        _visited_locations = set([str(loc.x) + '_' + str(loc.y)
                                  for loc in self.visited_locations])
        _inferred_pit_probs = set(
            [key for key, value in dict(self.inferred_pit_probs).items() if value < tolerance])

        _inferred_wumpus_probs = set(
            [key for key, value in dict(self.inferred_wumpus_probs).items() if value < 1./15])
        _safe_locations = _inferred_pit_probs.union(_visited_locations)
        _safe_locations = _inferred_wumpus_probs.union(_safe_locations)
        return _safe_locations

    def _get_breeze_cpt(self):
        neighbors = self._get_neighbors()
        n_neighbors = len(neighbors)
        n_neighbors += 1
        cpt = np.zeros(n_neighbors)

        for i in range(1, 2**n_neighbors):
            bin_rep = np.array([int(x)
                                for x in list(np.binary_repr(i, width=n_neighbors))])
            cpt = np.vstack((cpt, bin_rep))

        any_pit = np.max(cpt[:, :n_neighbors-1], axis=1)
        any_breeze = cpt[:, -1]

        _proba = np.expand_dims(np.invert(np.logical_xor(
            any_pit, any_breeze)).astype(float), axis=1)
        cpt = np.hstack((cpt, _proba))
        return cpt

    def _get_wumpus_cpt(self):
        neighbors = self._get_neighbors()
        n_neighbors = len(neighbors)
        n_neighbors += 1
        cpt = np.zeros(n_neighbors)

        for i in range(1, 2**n_neighbors):
            bin_rep = np.array([int(x)
                                for x in list(np.binary_repr(i, width=n_neighbors))])
            cpt = np.vstack((cpt, bin_rep))

        cpt_i = cpt[:, :n_neighbors-1]
        any_wumpus = np.count_nonzero(
            cpt_i == 1, axis=1)
        any_wumpus[any_wumpus > 1] = 0
        any_stench = cpt[:, -1]

        _proba = np.expand_dims(np.logical_and(
            any_wumpus, any_stench).astype(float), axis=1)
        cpt = np.hstack((cpt, _proba))

        # additional inputs that are needed
        cpt_wl, cells = [], []
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                curr_pos = str(self.agent_state.location.x) + \
                    '_' + str(self.agent_state.location.y)
                curr_cell = str(x) + "_" + str(y)
                cells.append(curr_cell)

                if curr_pos == curr_cell:
                    cpt_wl.append([curr_cell,  1.0, 1.0])
                    cpt_wl.append([curr_cell,  0.0, 0.0])
                else:
                    cpt_wl.append([curr_cell,  1.0, 0.0])
                    cpt_wl.append([curr_cell,  0.0, 1.0])

        return cpt_wl, cpt, set(cells)

    def _get_breeze_model(self):
        neighbors = self._get_neighbors()
        pits, nodes, breezes = {}, {}, {}

        for n_loc in neighbors:
            pits[n_loc] = DiscreteDistribution(
                {1: self.pit_proba, 0: 1 - self.pit_proba})
            nodes[n_loc] = State(pits[n_loc], name=n_loc)

        loc = str(self.agent_state.location.x)+'_' + \
            str(self.agent_state.location.y)
        breezes[loc] = ConditionalProbabilityTable(
            self._get_breeze_cpt(), [pits[n_loc] for n_loc in neighbors])
        nodes[loc] = State(breezes[loc], name=loc)

        model = BayesianNetwork("pits and breezes")
        for state in nodes:
            model.add_states(nodes[state])
        for pit in list(pits.keys()):
            for breeze in list(breezes.keys()):
                model.add_edge(nodes[pit], nodes[breeze])
        model.bake()
        return model

    def _get_wumpus_model(self):
        cpt_0, cpt_1, cells = self._get_wumpus_cpt()
        neighbors = self._get_neighbors()
        wumpuses, nodes, stenches = {}, {}, {}
        wumpus_location = DiscreteDistribution(
            {i: 1./((self.grid_width * self.grid_height) - 1) for i in cells})

        nodes['wumpus_location'] = State(
            wumpus_location, name='wumpus_location')
        for n_loc in neighbors:
            wumpuses[n_loc] = ConditionalProbabilityTable(
                cpt_0, [wumpus_location])
            nodes[n_loc] = State(wumpuses[n_loc], name=n_loc)

        loc = str(self.agent_state.location.x)+'_' + \
            str(self.agent_state.location.y)
        stenches[loc] = ConditionalProbabilityTable(
            cpt_1, [wumpuses[n_loc] for n_loc in neighbors])
        nodes[loc] = State(stenches[loc], name=loc)
        model = BayesianNetwork("wumpuses and stenches")
        for state in nodes:
            model.add_states(nodes[state])
        for wumpus in list(wumpuses.keys()):
            for stench in list(stenches.keys()):
                model.add_edge(nodes['wumpus_location'], nodes[wumpus])
                model.add_edge(nodes[wumpus], nodes[stench])
        model.bake()
        return model

    def _get_pit_post_proba(self, percept):
        loc = str(self.agent_state.location.x)+'_' + \
            str(self.agent_state.location.y)

        model = self._get_breeze_model()
        safe_locations = self._get_safe_locations()
        state_names = [state.name for state in model.states]
        neighbors = self._get_neighbors()

        input_neighbors = {
            item: 0 for item in state_names if (item in list(safe_locations)) & (item not in [loc])}

        if percept.breeze:
            input_neighbors[loc] = 1
        else:
            input_neighbors[loc] = 0

        inferred_pit_probs = model.predict_proba([input_neighbors])[0]

        inferred_pit_probs = [{neighbors[i]: round(x.parameters[0].get(1), 2)}
                              for i, x in enumerate(inferred_pit_probs) if isinstance(x, DiscreteDistribution)]
        inferred_pit_probs = dict((key, val)
                                  for k in inferred_pit_probs for key, val in k.items())

        for loc in list(safe_locations):
            if loc in neighbors:
                inferred_pit_probs[loc] = 0.0

        print(f"inferred pit probabilities >> {inferred_pit_probs}")
        return inferred_pit_probs

    def _get_wumpus_post_proba(self, percept):
        loc = str(self.agent_state.location.x)+'_' + \
            str(self.agent_state.location.y)

        model = self._get_wumpus_model()
        safe_locations = self._get_safe_locations()
        state_names = [state.name for state in model.states]

        neighbors = self._get_neighbors()

        input_neighbors = {
            item: 0 for item in state_names if (item in list(safe_locations)) & (item not in [loc])}

        if percept.stench:
            input_neighbors[loc] = 1
        else:
            input_neighbors[loc] = 0

        inferred_wumpus_probs = model.predict_proba([input_neighbors])[0]

        inferred_wumpus_probs = inferred_wumpus_probs[0].parameters

        inferred_wumpus_probs = dict((key, np.round(val, 2))
                                     for k in inferred_wumpus_probs for key, val in k.items())

        # for loc in list(safe_locations):
        #     if loc in neighbors:
        #         inferred_wumpus_probs[loc] = 0.0

        print(f"inferred wumpus probabilities >> {inferred_wumpus_probs}")
        return inferred_wumpus_probs

    def _construct_plan_from_forward_path(self, forward_path):
        active_orientation = self.agent_state.orientation.orientation
        actions = []
        cur_pos_x,  cur_pos_y = self.agent_state.location.x, self.agent_state.location.y

        for node in forward_path:
            nxt_pos_x,  nxt_pos_y = node  # reversed?
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

        return actions

    def get_forward_plan(self, inferred_pit_probs, inferred_wumpus_probs):
        neighborhood_proba = inferred_pit_probs.copy()
        neighbors = self._get_neighbors()
        neighborhood_proba = {n: neighborhood_proba[n]
                              for n in neighbors}

        if not neighborhood_proba:
            forward_path = tuple(int(i)
                                 for i in random.choice(neighbors).split('_'))
        else:
            neighborhood_choice = neighborhood_proba
            _rand_number = np.random.randint(low=1, high=10)
            if _rand_number > 2:
                try:
                    neighborhood_choice = dict((key, value)
                                               for key, value in neighborhood_proba.items() if value < 0.2)
                except:
                    neighborhood_choice = neighborhood_proba
                try:
                    forward_path = tuple(int(i) for i in random.choice(
                        list(neighborhood_choice.keys())).split('_'))
                except:
                    pass
            forward_path = tuple(int(i)
                                 for i in random.choice(neighbors).split('_'))
        forward_plan = self._construct_plan_from_forward_path([forward_path])
        return forward_plan

    def next_action(self, percept):
        visiting_new_location = not(any([(self.agent_state.location.x == loc.x) & (
            self.agent_state.location.y == loc.y) for loc in self.visited_locations]))
        new_visited_locations = self.visited_locations.copy()
        new_visited_locations.add(
            self.agent_state.location)
        new_breeze_locations = self.breeze_locations.copy()

        if percept.breeze:
            new_breeze_locations.add(
                self.agent_state.location)
        new_stench_locations = self.stench_locations.copy()
        if percept.stench:
            new_stench_locations.add(
                self.agent_state.location)
        new_heard_scream = self.heard_scream | percept.scream
        new_inferred_pit_probs = self.inferred_pit_probs.copy()
        new_inferred_wumpus_probs = self.inferred_wumpus_probs.copy()

        if (visiting_new_location) & (not(percept.glitter)):
            new_inferred_pit_probs.update(self._get_pit_post_proba(percept))
            # if not(new_heard_scream):
            new_inferred_wumpus_probs.update(self._get_wumpus_post_proba(
                percept))

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
                new_agent = self.__copy__()
                new_agent.agent_state = new_agent.agent_state.apply_move_action(
                    _action, self.grid_width, self.grid_height
                )
                new_agent.beeline_action_list = _beeline_plan[1:]

        elif percept.glitter:
            new_agent = self.__copy__()
            new_agent.agent_state.has_gold = True
            _action = Action.grab
        elif (percept.stench) & (self.agent_state.has_arrow):
            new_agent = self.__copy__()
            new_agent.agent_state.has_arrow = False
            _action = Action.shoot
        else:
            _beeline_plan = (
                self.get_forward_plan(
                    new_inferred_pit_probs, new_inferred_wumpus_probs)
                if not self.beeline_action_list
                else self.beeline_action_list
            )
            _action = _beeline_plan[0]
            new_agent = self.__copy__()
            new_agent.agent_state = new_agent.agent_state.apply_move_action(
                _action, self.grid_width, self.grid_height
            )
            new_agent.beeline_action_list = _beeline_plan[1:]
            # _rand_number = np.random.randint(low=1, high=5)
            # if _action == Action.forward:
            #     _new_agent_state = new_agent.agent_state.forward(
            #         self.grid_width, self.grid_height
            #     )
            #     new_agent.agent_state = _new_agent_state
            # elif _action == Action.turn_left:
            #     new_agent.agent_state = new_agent.agent_state.turn_left()
            # elif _action == Action.turn_right:
            #     new_agent.agent_state = new_agent.agent_state.turn_right()
            # elif _action == Action.shoot:
            #     new_agent.agent_state.use_arrow()

        new_agent.visited_locations = new_visited_locations
        new_agent.breeze_locations = new_breeze_locations
        new_agent.stench_locations = new_stench_locations
        new_agent.heard_scream = new_heard_scream
        new_agent.inferred_pit_probs = new_inferred_pit_probs
        new_agent.inferred_wumpus_probs = new_inferred_wumpus_probs
        return new_agent, _action


def initialize_probabilistic_agent(grid_width, grid_height):
    return ProbabilisticAgent(grid_width, grid_height, AgentState(), visited_locations=set(), beeline_action_list=[], breeze_locations=set(), stench_locations=set(), heard_scream=False, pit_proba=0.2, inferred_pit_probs=dict(), inferred_wumpus_probs=dict())
