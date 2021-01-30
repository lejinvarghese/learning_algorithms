from environment import Action, Percept, Orientation
from numpy.random import randint

action = Action(randint(1,6))
print(action)

percept = Percept().show()
print(percept)

orient = Orientation()
print(orient.active_orientation)
orient.turn_left()
print(orient.active_orientation)
orient.turn_left()
print(orient.active_orientation)
orient.turn_left()
print(orient.active_orientation)
orient.turn_right()
print(orient.active_orientation)