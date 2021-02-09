from environment.environment import Action, Percept, Environment, Orientation, Direction
from environment.agent import Agent
# from agent.agent import NaiveAgent


def main():

    def run_episode(environment, agent, percept):
        print(agent.orientation.orientation)
        next_action = agent.next_action(percept)
        print(f"next action: {next_action}")
        next_environment, next_percept = environment.apply_action(
            next_action)

        print(next_environment.visualize())
        print(next_percept.show())
        next_percept.reward += run_episode(next_environment, agent,
                                           next_percept) if not(next_percept.is_terminated) else 0.0
        return next_percept.reward

    naive_agent = Agent(orientation=Orientation(Direction.east))

    initial_environment, initial_percept = Environment(grid_width=5, grid_height=5, pit_proba=0.2, allow_climb_without_gold=True,
                                                       agent=naive_agent, pit_locations=None, terminated=False, wumpus_location=None, wumpus_alive=True, gold_location=None).initialize()

    # for i in range(5):
    total_reward = run_episode(
        initial_environment, naive_agent, initial_percept)


if __name__ == "__main__":
    main()
