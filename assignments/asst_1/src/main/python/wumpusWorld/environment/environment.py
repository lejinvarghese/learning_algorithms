#!/usr/bin/env python3

from enum import Enum
from numpy.random import randint

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

    def __init__(self, stench = False, breeze = False, glitter = False, bump = False, scream = False, is_terminated = False, reward = 0.0):
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

    def __init__(self, orientation = Direction.east):
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

class Environment (object):
    
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
        return any(x.x == location.x & x.y == location.y for x in self.pit_locations)

    def _is_wumpus_at(self, location):
        return (self.wumpus_location.x == location.x & self.wumpus_location.y == location.y)

    def _is_agent_at(self, location):
        return (self.agent.location.x == location.x & self.agent.location.y == location.y)

    def _is_glitter(self):
        return (self.gold_location.x == self.gold_location.x & self.gold_location.x == self.gold_location.x )

    def _is_gold_at(self, location):
        return (self.gold_location.x == location.x & self.gold_location.y == location.y)

    def _kill_attempt_successful(self):

        def _wumpus_in_line_of_fire(self):
            orientation_value = self.agent.orientation.orientation
            if orientation_value == Direction.west:
                return (self.agent.location.x > self.wumpus_location.x & self.agent.location.y == self.wumpus_location.y)
            elif orientation_value == Direction.east:
                return (self.agent.location.x < self.wumpus_location.x & self.agent.location.y == self.wumpus_location.y)
            elif orientation_value == Direction.south:
                return (self.agent.location.x == self.wumpus_location.x & self.agent.location.y > self.wumpus_location.y)
            elif orientation_value == Direction.north:
                return (self.agent.location.x == self.wumpus_location.x & self.agent.location.y < self.wumpus_location.y)
        
        return (self.agent.has_arrow & self.wumpus_alive & _wumpus_in_line_of_fire(self))

    def _adjacent_cells(self, location):

        _left = Coordinates(location.x -1, location.y) if location.x>0 else None
        _right = Coordinates(location.x + 1, location.y) if location.x < (self.grid_width - 1) else None
        _below = Coordinates(location.x, location.y - 1) if location.y>0 else None
        _above = Coordinates(location.x, location.y + 1) if location.y < (self.grid_height - 1) else None
        return [_left, _right, _below, _above]

    def _is_pit_adjacent(self, location):
        return any(x in self.pit_locations for x in self._adjacent_cells(location))

    def _is_wumpus_adjacent(self, location):
        return self.wumpus_location in self._adjacent_cells(location)

    def _is_breeze(self):
        return self._is_pit_adjacent(self.agent.location)

    def _is_stench(self):
        return self._is_wumpus_adjacent(self.agent.location) | self._is_wumpus_at(self.agent.location)

# final case class Environment private(
#                                       gridWidth: Int,
#                                       gridHeight: Int,
#                                       pitProb: Probability,
#                                       allowClimbWithoutGold: Boolean,
#                                       agent: Agent,
#                                       pitLocations: List[Coords],
#                                       terminated: Boolean,
#                                       wumpusLocation: Coords,
#                                       wumpusAlive: Boolean,
#                                       goldLocation: Coords
#                                     ) {
#   private def isPitAt(coords: Coords): Boolean = pitLocations.contains(coords)

#   private def isWumpusAt(coords: Coords): Boolean = wumpusLocation == coords

#   private def isAgentAt(coords: Coords): Boolean = coords == agent.location

#   private def isGlitter: Boolean = goldLocation == agent.location

#   private def isGoldAt(coords: Coords): Boolean = coords == goldLocation

#   private def killAttemptSuccessful: Boolean = {
#     def wumpusInLineOfFire: Boolean = {
#       agent.orientation match {
#         case West => agent.location.x > wumpusLocation.x && agent.location.y == wumpusLocation.y
#         case East => agent.location.x < wumpusLocation.x && agent.location.y == wumpusLocation.y
#         case South => agent.location.x == wumpusLocation.x && agent.location.y > wumpusLocation.y
#         case North => agent.location.x == wumpusLocation.x && agent.location.y < wumpusLocation.y
#       }
#     }

#     agent.hasArrow && wumpusAlive && wumpusInLineOfFire
#   }

#   private def adjacentCells(coords: Coords): List[Coords] = {
#     val toLeft: List[Coords] = if (coords.x > 0) List(Coords(coords.x - 1, coords.y)) else Nil

#     val toRight: List[Coords] = if (coords.x < gridWidth - 1) List(Coords(coords.x + 1, coords.y)) else Nil

#     val below: List[Coords] = if (coords.y > 0) List(Coords(coords.x, coords.y - 1)) else Nil

#     val above: List[Coords] = if (coords.y < gridHeight - 1) List(Coords(coords.x, coords.y + 1)) else Nil

#     toLeft ::: toRight ::: below ::: above
#   }

#   private def isPitAdjacent(coords: Coords): Boolean = {
#     adjacentCells(coords).exists(cell => pitLocations.contains(cell))
#   }

#   private def isWumpusAdjacent(coords: Coords): Boolean = {
#     adjacentCells(coords).exists(cell => isWumpusAt(cell))
#   }

#   private def isBreeze: Boolean = isPitAdjacent(agent.location)

#   private def isStench: Boolean = isWumpusAdjacent(agent.location) || isWumpusAt(agent.location)

#   def applyAction(action: Action): (Environment, Percept) = {

#     if (terminated)
#       (
#         this,
#         Percept(false, false, false, false, false, true, 0)
#       )
#     else {
#       action match {
#         case Forward =>
#           val movedAgent = agent.forward(gridWidth, gridHeight)
#           val death = (isWumpusAt(movedAgent.location) && wumpusAlive) || isPitAt(movedAgent.location)
#           val newAgent = movedAgent.copy(isAlive = !death)
#           val newEnv = new Environment(gridWidth, gridHeight, pitProb, allowClimbWithoutGold, newAgent, pitLocations, death, wumpusLocation, wumpusAlive,  if (agent.hasGold) newAgent.location else goldLocation)
#           (
#             newEnv,
#             Percept(newEnv.isStench, newEnv.isBreeze, newEnv.isGlitter, newAgent.location == agent.location, false, !newAgent.isAlive, if (newAgent.isAlive) -1 else -1001)
#           )
#         case TurnLeft =>
#           (
#             new Environment(gridWidth, gridHeight, pitProb, allowClimbWithoutGold, agent.turnLeft, pitLocations, terminated, wumpusLocation, wumpusAlive, goldLocation),
#             Percept(isStench, isBreeze, isGlitter,false, false,  false, -1)
#           )
#         case TurnRight =>
#           (
#             new Environment(gridWidth, gridHeight, pitProb, allowClimbWithoutGold, agent.turnRight, pitLocations, terminated, wumpusLocation, wumpusAlive, goldLocation),
#             Percept(isStench, isBreeze, isGlitter,false, false,  false, -1)
#           )
#         case Grab =>
#           val newAgent = agent.copy(hasGold = isGlitter)
#           (
#             new Environment(gridWidth, gridHeight, pitProb, allowClimbWithoutGold, newAgent, pitLocations, terminated, wumpusLocation, wumpusAlive, if (newAgent.hasGold) agent.location else goldLocation),
#             Percept(isStench, isBreeze, isGlitter,false, false,  false, -1)
#           )
#         case Climb =>
#           val inStartLocation = agent.location == Coords(0, 0)
#           val success = agent.hasGold && inStartLocation
#           val isTerminated = success || (allowClimbWithoutGold && inStartLocation)
#           (
#             new Environment(gridWidth, gridHeight, pitProb, allowClimbWithoutGold, agent, pitLocations, isTerminated, wumpusLocation, wumpusAlive, goldLocation),
#             Percept(false, false, isGlitter, false, false, isTerminated, if (success) 999 else -1)
#           )
#         case Shoot =>
#           val hadArrow = agent.hasArrow
#           val wumpusKilled = killAttemptSuccessful
#           (
#             new Environment(gridWidth, gridHeight, pitProb, allowClimbWithoutGold, agent.copy(hasArrow = false), pitLocations, terminated, wumpusLocation, wumpusAlive && !wumpusKilled, goldLocation),
#             Percept(isStench, isBreeze, isGlitter, false, wumpusKilled, false, if (hadArrow) -11 else -1)
#           )
#       }
#     }
#   }

#   def visualize: String = {
#     val wumpusSymbol = if (wumpusAlive) "W" else "w"
#     val rows = for {
#       y <- gridHeight - 1 to 0 by -1
#       cells = for {
#         x <- 0 until gridWidth
#         c = s"${if (isAgentAt(Coords(x, y))) "A" else " "}${if (isPitAt(Coords(x, y))) "P" else " "}${if (isGoldAt(Coords(x, y))) "G" else " "}${if (isWumpusAt(Coords(x, y))) wumpusSymbol else " "}"
#       } yield c
#       row = cells.mkString("|")
#     } yield row
#     rows.mkString("\n")
#   }
# }

# object Environment {
#   private val randGen = new Random()

#   def apply(
#              gridWidth: Int,
#              gridHeight: Int,
#              pitProb: Probability,
#              allowClimbWithoutGold: Boolean
#            ): (Environment, Percept) = {

#     def randomLocationExceptOrigin: Coords = {
#       val x = randGen.nextInt(gridWidth)
#       val y = randGen.nextInt(gridHeight)
#       if (x == 0 && y == 0) // try again if (0, 0)
#         randomLocationExceptOrigin
#       else
#         Coords(x, y)
#     }

#     val pitLocations: List[Coords] = {
#       val cellIndexes =
#         for (x <- 0 until gridWidth; y <- 0 until gridHeight)
#           yield Coords(x, y)
#       cellIndexes.tail.filter(_ => randGen.nextFloat < pitProb.value) // tail removes (0, 0)
#     }.toList

#     val env = new Environment(
#       gridWidth,
#       gridHeight,
#       pitProb,
#       allowClimbWithoutGold,
#       Agent(),
#       pitLocations,
#       false,
#       randomLocationExceptOrigin,
#       true,
#       randomLocationExceptOrigin
#     )

#     (
#       env,
#       Percept(env.isStench, env.isBreeze, false,false, false,  false, 0.0)
#     )
#   }
# }
