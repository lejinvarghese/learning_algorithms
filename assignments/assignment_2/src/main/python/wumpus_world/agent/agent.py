#!/usr/bin/env python3
from environment.environment import AgentState, Action, Coordinates
import numpy as np


class Agent:

    def next_action(self, percept):
        return Action()


class NaiveAgent:

    def next_action(self, percept):
        _rand_action = np.random.randint(low=1, high=6)
        return Action(_rand_action)


class BeelineAgent:

    def __init__(self, grid_width, grid_height, agent_state, safe_locations, beeline_action_list):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.agent_state = agent_state
        self.safe_locations = safe_locations
        self.beeline_action_list = beeline_action_list

    def __copy__(self):
        return BeelineAgent(self.grid_width, self.grid_height, self.agent_state,
                            self.safe_locations, self.beeline_action_list)

    def _show(self):
        print(
            f'beeline agent state: {self.agent_state.show()} \n safe locations: {self.safe_locations} \n beeline action list: {self.beeline_action_list}')

    def _construct_beeline_plan(self):
        pass

    def next_action(self, percept):
        if self.agent_state.has_gold:
            if (self.agent_state.location.x == 0) & (self.agent_state.location.y == 0):
                new_agent = self
                _action = Action.climb
                return new_agent, _action
            else:
                _beeline_plan = self._construct_beeline_plan(
                ) if not self.beeline_action_list else self.beeline_action_list
                _action = _beeline_plan[0]
                new_agent = BeelineAgent.__copy__(self)
                new_agent.agent_state = self.agent_state.apply_move_action(
                    _action, self.grid_width, self.grid_height)
                new_agent.beeline_action_list = _beeline_plan[1:]
                return new_agent, _action

        elif percept.glitter:
            new_agent = BeelineAgent.__copy__(self)
            new_agent.has_gold = True
            _action = Action.grab
            return new_agent, _action
        else:
            _rand_number = np.random.randint(low=1, high=4)
            if _rand_number == 1:
                new_agent = BeelineAgent.__copy__(self)
                _new_agent_state = new_agent.agent_state.forward(
                    self.grid_width, self.grid_height)
                _new_safe_locations_l = list(self.safe_locations)
                _new_safe_locations_l.extend([_new_agent_state.location])
                new_agent.agent_state = _new_agent_state
                new_agent.safe_locations = set(_new_safe_locations_l)
                _action = Action.forward
                return new_agent, _action
            elif _rand_number == 2:
                new_agent = BeelineAgent.__copy__(self)
                new_agent.agent_state = new_agent.agent_state.turn_left()
                _action = Action.turn_left
                return new_agent, _action
            elif _rand_number == 3:
                _new_agent_state = self.agent_state.turn_right()
                new_agent = BeelineAgent.__copy__(self)
                new_agent.agent_state = _new_agent_state
                _action = Action.turn_right
                return new_agent, _action
            elif _rand_number == 4:
                new_agent = BeelineAgent.__copy__(self)
                new_agent.agent_state.use_arrow()
                _action = Action.shoot
                return new_agent, _action


def initialize_beeline_agent(grid_width, grid_height):
    return BeelineAgent(grid_width, grid_height, AgentState(), set(), [])
