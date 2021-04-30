from IPython import display
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
from collections import deque

from gym import Env, spaces
import time

font = cv2.FONT_HERSHEY_DUPLEX
AGENT_SIZE = 32
ITEM_SIZE = 24
ITEM_PROBABILITY = 0.05


class MarioWorld(Env):
    def __init__(self):
        super(MarioWorld, self).__init__()

        # Define an observation space
        self.observation_shape = (600, 800, 3)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.ones(
                                                self.observation_shape),
                                            dtype=np.float16)

        # Define an action space
        self.action_space = spaces.Discrete(9,)

        # Create a canvas to render the environment images
        self.canvas = np.ones(self.observation_shape) * 1

        # Define elements present inside the environment
        self.elements = []

        # Maximum fuel mario can take at once
        self.max_fuel = 1000

        # Permissible area of helicopter to be
        self.y_min = int(self.observation_shape[0] * 0.05)
        self.x_min = 0
        self.y_max = int(self.observation_shape[0] * 0.95)
        self.x_max = self.observation_shape[1]

    def get_action_meanings(self, action):
        actions = {0: "East", 1: "West", 2: "South",
                   3: "North", 4: "North-East",  5: "North-West",  6: "South-East",  7: "South-West",  8: "Do Nothing"}
        return actions.get(action)

    def draw_elements_on_canvas(self):

        self.canvas = np.ones(self.observation_shape) * 1

        for elem in self.elements:
            elem_shape = elem.icon.shape
            x, y = elem.x, elem.y
            self.canvas[y: y + elem_shape[1], x:x + elem_shape[0]] = elem.icon

        # Add text info on the canvas
        text = f" Fuel: {self.fuel_left} | Rewards: {self.ep_return} | Lives: {self.life_count}"
        self.canvas = cv2.putText(self.canvas, text, (10, 20), font,
                                  0.6, (0, 0, 0), 1, cv2.LINE_AA)

    def reset(self):
        # Reset the fuel consumed
        self.fuel_left = self.max_fuel

        # Reset the reward
        self.ep_return = 0

        # Number of elements
        self.bullet_count = 0
        self.life_count = 1
        self.fuel_count = 0
        self.action_buffer = deque(maxlen=10)

        # Determine a place to intialise the mario in
        x = random.randrange(
            int(self.observation_shape[0] * 0.05), int(self.observation_shape[0] * 0.10))
        y = random.randrange(
            int(self.observation_shape[1] * 0.15), int(self.observation_shape[1] * 0.20))

        # Intialise the mario
        self.mario = Mario("mario", self.x_max,
                           self.x_min, self.y_max, self.y_min)
        self.mario.set_position(x, y)

        # Intialise the elements
        self.elements = [self.mario]

        # Reset the Canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # return the observation
        return self.canvas

    def render(self, mode="human"):
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()

    def step(self, action):
        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the fuel counter
        self.fuel_left -= 1

        # Reward for executing a step.
        reward = 2

        # apply the action to the mario
        if action == 0:
            self.mario.move(0, 5)
        elif action == 1:
            self.mario.move(0, -5)
        elif action == 2:
            self.mario.move(5, 0)
        elif action == 3:
            self.mario.move(-5, 0)
        elif action == 4:
            self.mario.move(5, 5)
        elif action == 5:
            self.mario.move(5, -5)
        elif action == 6:
            self.mario.move(-5, 5)
        elif action == 7:
            self.mario.move(-5, -5)
        elif action == 8:
            self.mario.move(random.randint(0, 2), random.randint(0, 2))

        self.action_buffer.append(action)
        if len(set(self.action_buffer)) == 1:
            reward -= 100

        # Spawn a bullet at the right edge with given probability
        if random.random() < ITEM_PROBABILITY:

            spawned_bullet = Bullet("bullet_{}".format(self.bullet_count),
                                    self.x_max, self.x_min, self.y_max, self.y_min)
            self.bullet_count += 1
            bullet_x = self.x_max
            bullet_y = random.randrange(self.y_min, self.y_max)
            spawned_bullet.set_position(bullet_x, bullet_y)
            self.elements.append(spawned_bullet)

        # Spawn fuel at the bottom edge with given probability ITEM_PROBABILITY
        if random.random() < ITEM_PROBABILITY/2:
            spawned_fuel = Fuel("fuel_{}".format(self.bullet_count),
                                self.x_max, self.x_min, self.y_max, self.y_min)
            self.fuel_count += 1
            fuel_x = random.randrange(self.x_min, self.x_max)
            fuel_y = self.y_max
            spawned_fuel.set_position(fuel_x, fuel_y)
            self.elements.append(spawned_fuel)

        # Spawn  life at the bottom edge with given probability ITEM_PROBABILITY
        if random.random() < ITEM_PROBABILITY/4:
            spawned_life = Life("life_{}".format(self.life_count),
                                self.x_max, self.x_min, self.y_max, self.y_min)
            life_x = random.randrange(self.x_min, self.x_max)
            life_y = self.y_max
            spawned_life.set_position(life_x, life_y)
            self.elements.append(spawned_life)

        for elem in self.elements:
            # If the bullet has reached the left end, perform these actions
            if isinstance(elem, Bullet):
                if elem.get_position()[0] <= self.x_min:
                    self.elements.remove(elem)
                else:
                    elem.move(-5, 0)

                if self.has_collided(self.mario, elem):
                    self.life_count -= 1
                    if self.life_count < 1:
                        reward = -400
                        done = True
                        self.elements.remove(self.mario)
                    reward += -100

            if isinstance(elem, Fuel):
                # If the fuel has reached the top end, perform these actions
                if elem.get_position()[1] <= self.y_min:
                    self.elements.remove(elem)
                else:
                    elem.move(0, -5)

                if self.has_collided(self.mario, elem):
                    self.elements.remove(elem)
                    self.fuel_left = self.max_fuel
                    reward += 400

            if isinstance(elem, Life):
                # If the life has reached the top end, perform these actions
                if elem.get_position()[1] <= self.y_min:
                    self.elements.remove(elem)
                else:
                    elem.move(0, -5)

                if self.has_collided(self.mario, elem):
                    self.elements.remove(elem)
                    self.life_count += 10

        self.ep_return += 1 + reward + self.life_count * 5

        self.draw_elements_on_canvas()

        if self.fuel_left == 0:
            done = True
        return self.canvas, reward,  done, {}

    def has_collided(self, elem1, elem2):

        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
            if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
                return True

        return False


class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


class Mario(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Mario, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("assets/mario.jpg") / 255.
        self.icon_w = AGENT_SIZE
        self.icon_h = AGENT_SIZE
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Bullet(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Bullet, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread(
            "assets/bullet.png") / 255.
        self.icon_w = ITEM_SIZE
        self.icon_h = ITEM_SIZE
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Fuel(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Fuel, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread(
            "assets/fuel.jpg") / 255.
        self.icon_w = ITEM_SIZE
        self.icon_h = ITEM_SIZE
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Life(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Life, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread(
            "assets/life.jpg") / 255.
        self.icon_w = ITEM_SIZE
        self.icon_h = ITEM_SIZE
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


env = MarioWorld()
obs = env.reset()

while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    # Render the game
    env.render()
    if done == True:
        break

env.close()
