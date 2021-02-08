from environment import Action, Percept, Orientation, Coordinates, Direction, Environment
from numpy.random import randint
from agent import Agent

action = Action(randint(1,6))
print(action)

print(Direction(2), Direction(1))

percept = Percept().show()
print(percept)

# orient = Orientation()
# print(orient.orientation)
# orient.turn_left()
# print(orient.orientation)


locations = [Coordinates(0, 0), Coordinates(1, 0)]

print('cond', locations[0].x == 0 & locations[0].y ==0)



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
