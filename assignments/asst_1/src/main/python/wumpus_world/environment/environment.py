#!/usr/bin/env python3

from enum import Enum
import numpy as np


class Action(Enum):

    forward = 1
    turn_left = 2
    turn_right = 3
    shoot = 4
    grab = 5
    climb = 6

    def __init__(self, percept):
        self.percept = percept


class Percept:

    def __init__(self, stench=False, breeze=False, glitter=False, bump=False, scream=False, is_terminated=False, reward=0.0):
        self.stench = stench
        self.breeze = breeze
        self.glitter = glitter
        self.bump = bump
        self.scream = scream
        self.is_terminated = is_terminated
        self.reward = reward

    def show(self):
        print(f'stench: {self.stench} \n breeze: {self.breeze} \n glitter: {self.glitter} \n bump: {self.bump} \n scream: {self.scream} \n is_terminated: {self.is_terminated} \n reward: {self.reward}')


class Direction(Enum):
    north = 1
    south = 2
    east = 3
    west = 4


class Orientation:

    def __init__(self, orientation=Direction.east):
        self.orientation = orientation

    def turn_left(self):
        if self.orientation == Direction.north:
            self.orientation = Direction.west
        elif self.orientation == Direction.south:
            self.orientation = Direction.east
        elif self.orientation == Direction.east:
            self.orientation = Direction.north
        elif self.orientation == Direction.west:
            self.orientation = Direction.south

    def turn_right(self):
        if self.orientation == Direction.north:
            self.orientation = Direction.east
        elif self.orientation == Direction.south:
            self.orientation = Direction.west
        elif self.orientation == Direction.east:
            self.orientation = Direction.south
        elif self.orientation == Direction.west:
            self.orientation = Direction.north


class Coordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Environment:

    def __init__(self, grid_width, grid_height, pit_proba, allow_climb_without_gold, agent, pit_locations, terminated, wumpus_location, wumpus_alive, gold_location):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.pit_proba = pit_proba
        self.allow_climb_without_gold = allow_climb_without_gold
        self.agent = agent
        self.pit_locations = pit_locations
        self.terminated = terminated
        self.wumpus_location = wumpus_location
        self.wumpus_alive = wumpus_alive
        self.gold_location = gold_location

    def _is_pit_at(self, location):
        return any((pit.x == location.x) & (pit.y == location.y) for pit in self.pit_locations)

    def _is_wumpus_at(self, location):
        return (self.wumpus_location.x == location.x) & (self.wumpus_location.y == location.y)

    def _is_agent_at(self, location):
        return (self.agent.location.x == location.x) & (self.agent.location.y == location.y)

    def _is_glitter(self):
        return (self.gold_location.x == self.agent.location.x) & (self.gold_location.y == self.agent.location.y)

    def _is_gold_at(self, location):
        return (self.gold_location.x == location.x) & (self.gold_location.y == location.y)

    def _kill_attempt_successful(self):

        def _wumpus_in_line_of_fire(self):
            orientation_value = self.agent.orientation.orientation
            if orientation_value == Direction.west:
                return (self.agent.location.x > self.wumpus_location.x) & (self.agent.location.y == self.wumpus_location.y)
            elif orientation_value == Direction.east:
                return (self.agent.location.x < self.wumpus_location.x) & (self.agent.location.y == self.wumpus_location.y)
            elif orientation_value == Direction.south:
                return (self.agent.location.x == self.wumpus_location.x) & (self.agent.location.y > self.wumpus_location.y)
            elif orientation_value == Direction.north:
                return (self.agent.location.x == self.wumpus_location.x) & (self.agent.location.y < self.wumpus_location.y)

        return (self.agent.has_arrow) & (self.wumpus_alive) & (_wumpus_in_line_of_fire(self))

    def _adjacent_cells(self, location):

        _left = Coordinates(
            location.x - 1, location.y) if location.x > 0 else None
        _right = Coordinates(
            location.x + 1, location.y) if location.x < (self.grid_width - 1) else None
        _below = Coordinates(location.x, location.y -
                             1) if location.y > 0 else None
        _above = Coordinates(location.x, location.y +
                             1) if location.y < (self.grid_height - 1) else None
        return [_left, _right, _below, _above]

    def _is_pit_adjacent(self, location):
        return any(x in self.pit_locations for x in self._adjacent_cells(location))

    def _is_wumpus_adjacent(self, location):
        return self.wumpus_location in self._adjacent_cells(location)

    def _is_breeze(self):
        return self._is_pit_adjacent(self.agent.location)

    def _is_stench(self):
        return self._is_wumpus_adjacent(self.agent.location) | self._is_wumpus_at(self.agent.location)

    def apply_action(self, action):

        if self.terminated:
            return Percept(False, False, False, False, False, True, 0)
        else:
            if action == Action.forward:
                moved_agent = self.agent.forward(
                    self.grid_width, self.grid_height)
                death = (self.wumpus_alive & self._is_wumpus_at(moved_agent.location)) | (
                    self._is_pit_at(moved_agent.location))
                new_agent = moved_agent.__copy__()
                new_agent.is_alive = not death
                new_environment = Environment(
                    self.grid_width, self.grid_height, self.pit_proba, self.allow_climb_without_gold, new_agent, self.pit_locations, death, self.wumpus_location, self.wumpus_alive, new_agent.location if self.agent.has_gold else self.gold_location)
                new_percept = Percept(new_environment._is_stench(), new_environment._is_breeze(), new_environment._is_glitter(),
                                      new_agent.location == self.agent.location, False, not new_agent.is_alive, -1 if new_agent.is_alive else -1001)
                return (new_environment, new_percept)
            elif action == Action.turn_left:
                new_environment = Environment(
                    self.grid_width, self.grid_height, self.pit_proba, self.allow_climb_without_gold, self.agent.turn_left(), self.pit_locations, self.terminated, self.wumpus_location, self.wumpus_alive, self.gold_location)
                new_percept = Percept(self._is_stench(), self._is_breeze(), self._is_glitter(),
                                      False, False, False, -1)
                return (new_environment, new_percept)
            elif action == Action.turn_right:
                new_environment = Environment(
                    self.grid_width, self.grid_height, self.pit_proba, self.allow_climb_without_gold, self.agent.turn_right(), self.pit_locations, self.terminated, self.wumpus_location, self.wumpus_alive, self.gold_location)
                new_percept = Percept(self._is_stench(), self._is_breeze(), self._is_glitter(),
                                      False, False, False, -1)
                return (new_environment, new_percept)
            elif action == Action.grab:
                new_agent = self.agent.__copy__()
                new_agent.has_gold = self._is_glitter()
                new_environment = Environment(
                    self.grid_width, self.grid_height, self.pit_proba, self.allow_climb_without_gold, new_agent, self.pit_locations, self.terminated, self.wumpus_location, self.wumpus_alive, self.agent.location if new_agent.has_gold else self.gold_location)
                new_percept = Percept(self._is_stench(), self._is_breeze(), self._is_glitter(),
                                      False, False, False, -1)
                return (new_environment, new_percept)
            elif action == Action.climb:
                in_start_location = self.agent.location == Coordinates(0, 0)
                success = self.agent.has_gold & in_start_location
                is_terminated = success | (
                    self.allow_climb_without_gold & in_start_location)
                new_environment = Environment(
                    self.grid_width, self.grid_height, self.pit_proba, self.allow_climb_without_gold, self.agent, self.pit_locations, self.terminated, self.wumpus_location, self.wumpus_alive, self.gold_location)
                new_percept = Percept(False, False, self._is_glitter(),
                                      False, False, is_terminated, 999 if success else -1)
                return (new_environment, new_percept)
            elif action == Action.shoot:
                had_arrow = self.agent.has_arrow
                wumpus_killed = self._kill_attempt_successful()
                new_agent = self.agent.__copy__()
                new_agent.has_arrow = False
                new_environment = Environment(
                    self.grid_width, self.grid_height, self.pit_proba, self.allow_climb_without_gold, new_agent, self.pit_locations, self.terminated, self.wumpus_location, self.wumpus_alive & (not wumpus_killed), self.gold_location)
                new_percept = Percept(self._is_stench(), self._is_breeze(), self._is_glitter(),
                                      False, wumpus_killed, False, -11 if had_arrow else -1)
                return (new_environment, new_percept)

    def visualize(self):
        wumpus_symbol = "\U0001F47E" if self.wumpus_alive else "\U0001F480"
        arr = np.full((self.grid_width, self.grid_height), "  ")

        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if self._is_agent_at(Coordinates(x, y)):
                    arr[x, y] = "\U0001F425"
                elif self._is_pit_at(Coordinates(x, y)):
                    arr[x, y] = "\U0001F573 "
                elif self._is_gold_at(Coordinates(x, y)):
                    arr[x, y] = "\U0001F947"
                elif self._is_wumpus_at(Coordinates(x, y)):
                    arr[x, y] = wumpus_symbol
                else:
                    "  "

        return np.array2string(np.flipud(arr), separator="|", formatter={'str_kind': lambda x: x})

    def initialize(self):

        def random_location_except_origin(self):
            x = np.random.randint(self.grid_width)
            y = np.random.randint(self.grid_height)
            if (x == 0) & (y == 0):
                return random_location_except_origin(self)
            else:
                return Coordinates(x, y)

        pit_locations = [Coordinates(x, y) if np.random.uniform() <
                         self. pit_proba else None for x in range(self.grid_width) for y in range(self.grid_height)]
        pit_locations[0] = None  # setting starting point to 0
        pit_locations = list(filter(None.__ne__, pit_locations))

        environment = Environment(
            self.grid_width, self.grid_height, self.pit_proba, self.allow_climb_without_gold, self.agent, pit_locations, False, random_location_except_origin(self), True, random_location_except_origin(self))

        percept = Percept(environment._is_stench(
        ), environment._is_breeze(), False, False, False, False, 0.0)

        return (environment, percept)
