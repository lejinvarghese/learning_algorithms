from environment.environment import initialize_environment
from agent.agent import initialize_beeline_agent


def main():

    def run_episode(environment, agent, percept):

        next_agent, next_action = agent.next_action(percept)
        print(f"agent next action: {next_action}")

        next_environment, next_percept = environment.apply_action(
            next_action)
        print(next_environment.visualize())
        print(next_percept.show())
        if not(next_percept.is_terminated):
            next_percept.reward += run_episode(next_environment, next_agent,
                                               next_percept)
        else:
            next_percept.reward += 0.0

        return next_percept.reward

    initial_environment, initial_percept = initialize_environment(
        grid_width=4, grid_height=4, pit_proba=0.0, allow_climb_without_gold=True)
    print("initial environment >>")
    print(initial_environment.visualize())

    agent = initialize_beeline_agent(4, 4)
    total_reward = run_episode(
        initial_environment, agent, initial_percept)
    print("total_reward: ", total_reward)


if __name__ == "__main__":
    main()
