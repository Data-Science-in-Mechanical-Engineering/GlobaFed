from bayesian_optimization_dp_fts_de import dp_fts_de
import pickle
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

max_iter = 50
dim = 10
M = 100
random_features = pickle.load(open("aux_files/RF_M_" +
                                   str(M) + "_dim_" + str(dim) + ".pkl", "rb"))

M_target = 100  # number of random features used to sample a function from an agent's own GP posterior

N = 10  # number of agents

pt = 1 - 1 / (np.arange(max_iter+5) + 1)


# Pre-generate a random shift per agent (in [-0.5, 0.5] for example)
np.random.seed(0)


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


# ### use the parameters below if you want to run FTS or DP-FTS (without DE); but make sure you create the partition files using the "create_partitions.py" script
# N_partitions = 1
# a, b = 1, 1
# T_max_decay = 100
# T_const = 100
# use the parameters below if you want to run FTS-DE or DP-FTS-DE (without DE)
N_partitions = 1024
a, b = 16, 1
T_max_decay = 30
T_const = 10
partition_info = pickle.load(open("aux_files/partitions_info_N_part_" + str(N_partitions) + "_a_" + str(a) + "_b_" + str(b) +
                                  "_T_decay_" + str(T_max_decay) + "_T_const_" + str(T_const) + "_adaptive.pkl", "rb"))

partitions = partition_info["partitions"]
partition_assignment = partition_info["partition_assignment"]
all_weights = partition_info["all_weights"]
N_partitions = len(partitions)
print(N_partitions)

RDP = False  # whether to use Renyi DP

# set NO_DP to True if you don't want to use DP, in which case you'll simply run the FTS-DE algorithm
NO_DP = True

q = 0.35
S = 22.0
z = 1.0

run_list = np.arange(0, 30)
sub_list = np.arange(0, N)

pool = ThreadPool(N)

for run_iter in run_list:
    shifts = np.random.uniform(-3, 3, size=(N))

    def obj_func_ackley(x, i):
        x = np.array(x)
        # Scale from [0, 1] → [-5, 10]
        x_scaled = -5 + x * (10 - (-5))  # or x * 15 - 5
        # Shift per agent
        x_shifted = x_scaled - shifts[i]
        # Maximize: take the negative of standard Ackley
        return -ackley(x_shifted)

    def parallel_bo_define(s):
        print("[defining BO for agent {0}]".format(s))
        if NO_DP:
            log_file_name = "results_dp_fts_de/agent_" + str(s) + "_run_" + str(run_iter) + \
                "_N_part_" + str(N_partitions) + ".p"
        else:
            log_file_name = "results_dp_fts_de/agent_" + str(s) + "_run_" + str(run_iter) + \
                "_N_part_" + str(N_partitions) + "_q_" + str(q) + \
                "_S_" + str(S) + "_z_" + str(z) + ".p"

        fts = dp_fts_de(f=obj_func_ackley, pbounds={f'x{j}': (0, 1) for j in range(dim)}, gp_opt_schedule=5,
                        log_file=log_file_name, M=M, N=N, random_features=random_features,
                        pt=pt, M_target=M_target, partitions=partitions, partition_assignment=partition_assignment[s])
        return fts
    all_bo = pool.map(parallel_bo_define, sub_list)

    all_w_t = None
    for itr in np.arange(max_iter):
        init_flag = itr == 0

        # run parallel processes for all agents
        def parallel_bo(s):
            global all_bo
            return all_bo[s].maximize(n_iter=max_iter, init_points=10, all_w_t=all_w_t, init_flag=init_flag, iter_fed=itr, agent_ind=s)
        all_w_nt = pool.map(parallel_bo, sub_list)
        all_w_nt = np.squeeze(np.array(all_w_nt))

        if NO_DP:
            all_w_t = []
            for p in range(N_partitions):
                ws = all_weights[itr][p]
                w_sum = np.zeros(M)
                for a_ind, w in enumerate(all_w_nt):
                    w_sum += ws[a_ind] * w
                w_t = w_sum
                all_w_t.append(w_t)
        else:

            if RDP is False:
                select_subset_id = []
                for n in sub_list:
                    if np.random.random() < q:
                        select_subset_id.append(n)
                selected_size = len(select_subset_id)
            else:
                select_subset_id = list(np.random.choice(
                    sub_list, int(N * q), replace=False))
                selected_size = len(select_subset_id)

            if selected_size == 0:
                # this is a degenerate case which happens with very small probability, for which we simply draw a random vector
                all_w_t = []
                for p in range(N_partitions):
                    all_w_t.append(np.random.random(M))
            else:
                S_part = S / np.sqrt(N_partitions)
                all_w_t = []
                for p in range(N_partitions):
                    ws = all_weights[itr][p]
                    w_sum = np.zeros(M)

                    for a_ind in select_subset_id:
                        w = all_w_nt[a_ind]

                        w_clipped = w / np.max([1, np.sqrt(np.sum(w**2)) / S_part])
                        w_sum += ws[a_ind] * w_clipped / q

                    noise_vec = np.random.normal(0, np.max(ws) * z * S / q, M)

                    w_t = w_sum + noise_vec
                    all_w_t.append(w_t)
