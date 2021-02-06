from environment import Action, Percept, Orientation
from numpy.random import randint
from agent import Agent
from environment import Coordinates, Direction

action = Action(randint(1,6))
print(action)

print(Direction(2), Direction(1))

percept = Percept().show()
print(percept)

# orient = Orientation()
# print(orient.orientation)
# orient.turn_left()
# print(orient.orientation)


# location = Coordinates(0, 0)

agent = Agent()
print(agent.orientation.orientation)
print(agent.location.x, agent.location.y)

agent.orientation.turn_left()
print(agent.orientation.orientation)
print(agent.location.x, agent.location.y)

for i in range(6):
    agent = agent.forward(4, 4)
    if i==3:
        agent.orientation.turn_right()
    print(agent.orientation.orientation)
    print(agent.location.x, agent.location.y)
