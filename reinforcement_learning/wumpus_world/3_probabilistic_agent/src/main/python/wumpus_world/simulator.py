from probabilistic_world import main
import numpy as np
from multiprocessing import Pool, cpu_count


n_simulations = 1000
n_success = 0
n_cores = cpu_count()
rewards = []


def main_parallel(_):
    return main()


pool = Pool(n_cores)
rewards = pool.map(main_parallel, range(n_simulations))
pool.close()
pool.join()
print(f'simulations: ', n_simulations, 'mean reward: ', {np.round(np.mean(rewards), 2)},
      'median reward: ', {np.median(rewards)})
