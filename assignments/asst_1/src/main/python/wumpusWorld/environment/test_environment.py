from environment import Action, Percept, Orientation
from numpy.random import randint
from agent import Agent

action = Action(randint(1,6))
print(action)

percept = Percept().show()
print(percept)

orient = Orientation()
print(orient.active_orientation)
orient.turn_left()
print(orient.active_orientation)


