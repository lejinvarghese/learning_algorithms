#!/usr/bin/env python3
from wumpusWorld.environment.environment import Action, Percept

# trait Agent {
#   def nextAction(percept: Percept): Action
# }

class Agent:
  def __init__(self):
      pass

  def next_action(self, percept):
    return Action(percept)

agent = Agent()
print(agent.next_action(Percept))