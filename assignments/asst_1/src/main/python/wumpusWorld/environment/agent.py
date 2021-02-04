# import scala.math.{max, min}
from environment import Coords, Direction

class Agent:
  def __init__(self, location, orientation, has_gold, has_arrow, is_alive):
    self.location = location
    self.orientation = orientation
    self.has_gold = has_gold
    self.has_arrow = has_arrow
    self.is_alive = is_alive

  def __copy__(self):
    return Agent(self.location, self.orientation,self.has_gold,
    self.has_arrow, self.is_alive)

  def turn_left (self):
    newAgent = Agent.__copy__(self)
    newAgent.orientation.turn_left()
    return newAgent
  
  def turn_right (self):
    newAgent = Agent.__copy__(self)
    newAgent.orientation.turn_right()
    return newAgent

  def forward(self, grid_width, grid_height):
    new_location = self.location
    if self.orientation == Direction.west:
      new_location = Coords(max(0, self.location.x - 1), self.location.y)
    elif self.orientation == Direction.east:
      new_location = Coords(min(grid_width -1, self.location.x + 1), self.location.y)
    elif self.orientation == Direction.south:
      new_location = Coords(max(0, self.location.y - 1), self.location.x)
    elif self.orientation == Direction.north:
      new_location = Coords(min(grid_height -1, self.location.y + 1), self.location.x)
    newAgent = Agent.__copy__(self)
    newAgent.location = new_location
     
