from environment.environment import Coordinates, Direction, Orientation, Action
import numpy as np


class Agent:

    def __init__(self, location=Coordinates(0, 0), orientation=Orientation(), has_gold=False, has_arrow=True, is_alive=True):
        self.location = location
        self.orientation = orientation
        self.has_gold = has_gold
        self.has_arrow = has_arrow
        self.is_alive = is_alive

    def next_action(self, percept):
        _rand_action = np.random.randint(low=1, high=6)
        return Action(_rand_action)

    def __copy__(self):
        return Agent(self.location, self.orientation, self.has_gold,
                     self.has_arrow, self.is_alive)

    def turn_left(self):
        newAgent = Agent.__copy__(self)
        newAgent.orientation.turn_left()
        return newAgent

    def turn_right(self):
        newAgent = Agent.__copy__(self)
        newAgent.orientation.turn_right()
        return newAgent

    def forward(self, grid_width, grid_height):
        new_location = self.location
        orientation_value = self.orientation.orientation
        if orientation_value == Direction.west:
            new_location = Coordinates(
                max(0, self.location.x - 1), self.location.y)
        elif orientation_value == Direction.east:
            new_location = Coordinates(
                min(grid_width - 1, self.location.x + 1), self.location.y)
        elif orientation_value == Direction.south:
            new_location = Coordinates(
                self.location.x, max(0, self.location.y - 1))
        elif orientation_value == Direction.north:
            new_location = Coordinates(self.location.x, min(
                grid_height - 1, self.location.y + 1))
        newAgent = Agent.__copy__(self)
        newAgent.location = new_location
        return newAgent
