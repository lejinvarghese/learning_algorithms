from environment.environment import initialize_environment, Action
from agent.agent import initialize_beeline_agent
import torch
import numpy as np
import random

action_set = {
    0: Action(1),
    1: Action(2),
    2: Action(3),
    3: Action(4),
    4: Action(5),
    5: Action(6),
}
epsilon = 0.0


def main():

    def run_episode(environment, agent, state, mov=0):
        mov += 1
        state_ = torch.from_numpy(state).float()
        y_pred = agent(state_).data.numpy()

        if (random.random() < epsilon):
            next_action = np.random.randint(0, 6)
        else:
            next_action = np.argmax(y_pred)
        next_action = action_set[next_action]
        environment.agent = environment.agent.apply_move_action(
            next_action, 4, 4)
        print(f"agent next action: {next_action}")

        next_environment, next_percept = environment.apply_action(
            next_action)
        next_state = initial_environment.agent.q_transform()
        print(next_environment.visualize())
        print(next_percept.show())
        if (not(next_percept.is_terminated)) & (mov < 50):
            next_percept.reward += run_episode(next_environment, agent,
                                               next_state, mov)

        else:
            next_percept.reward += 0.0

        return next_percept.reward

    grid_width, grid_height = (4, 4)
    initial_environment, _ = initialize_environment(
        grid_width=grid_width, grid_height=grid_height, pit_proba=0.2, allow_climb_without_gold=False)
    initial_state = initial_environment.agent.q_transform()
    print("initial environment >>")
    print(initial_environment.visualize())

    agent = torch.load("./models")
    agent.eval()
    total_reward = run_episode(
        initial_environment, agent, initial_state)
    print("total_reward: ", total_reward)
    return total_reward


if __name__ == "__main__":
    try:
        main()
    except:
        -1000
