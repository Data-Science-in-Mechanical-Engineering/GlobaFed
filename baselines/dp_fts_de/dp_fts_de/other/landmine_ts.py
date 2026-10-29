from bayesian_optimization_ts import TS
import pickle
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool


max_iter = 50
N = 10
dim = 10

run_list = np.arange(0, 30)

pool = ThreadPool(10)  # Number of threads

# Pre-generate a random shift per agent (in [-0.5, 0.5] for example)
np.random.seed(0)
shifts = np.random.uniform(-3, 3, size=(N))

# Ackley constants
ACK_A = 20
ACK_B = 0.2
ACK_C = 2 * np.pi


def ackley(x):
    x = np.array(x)
    dim = len(x)
    sum_sq_term = -ACK_B * np.sqrt(np.sum(x ** 2) / dim)
    cos_term = np.sum(np.cos(ACK_C * x)) / dim
    return -ACK_A * np.exp(sum_sq_term) - np.exp(cos_term) + ACK_A + np.e


for i in range(N):
    def obj_func_ackley(x):
        x = np.array(x)
        # Shift per agent
        x_shifted = x - shifts[i]
        # Maximize: take the negative of standard ackley (which is a minimization function)
        return -ackley(x_shifted)

    def parallel_runs(itr):
        log_file_name = f"results_ts/agent_{i}_iter_{itr}.p"
        bo_ts = TS(
            f=obj_func_ackley,
            pbounds={f'x{j}': (-5, 10) for j in range(dim)},
            gp_opt_schedule=5,
            log_file=log_file_name,
            M_target=100
        )
        bo_ts.maximize(n_iter=max_iter, init_points=dim)

    pool.map(parallel_runs, run_list)
