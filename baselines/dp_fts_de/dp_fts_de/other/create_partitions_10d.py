'''
This script creates the partitions information to be used for distributed exploration (DE).
To run FTS (without DE), simply set N_partitions_single_dim=1, a=1, T_const=100, T_max_decay=100.
'''
import numpy as np
import itertools
import pickle

# Number of agents
N = 10

# Domain bounds (same for all dimensions)
Domain = np.array([0, 1])
Domain_width = Domain[1] - Domain[0]

# Partitioning parameters
N_partitions_single_dim = 2  # number of regions per dimension
d = 10  # input dimension

# Generate all partitions for d-dimensional space
partitions = []
for idx_tuple in itertools.product(range(N_partitions_single_dim), repeat=d):
    bounds = []
    for dim_index in idx_tuple:
        low = Domain[0] + dim_index * Domain_width / N_partitions_single_dim
        high = Domain[0] + (dim_index + 1) * Domain_width / N_partitions_single_dim
        bounds.append([low, high])
    partitions.append(np.array(bounds))

# Number of partitions
N_partitions = len(partitions)  # = N_partitions_single_dim ** d

# Assign agents to partitions cyclically
partition_assignment = np.arange(N) % N_partitions

print("N_partitions: ", N_partitions)
print("Example partition (first one):", partitions[0])
print("partition_assignment: ", partition_assignment)

# Adaptive weighting parameters
T_max = 200
a, b = 16, 1

# 'a' stays constant for T_const iterations, then decays linearly to b over T_max_decay iterations
T_const = 10
T_max_decay = 30

a_s = np.linspace(a, b, T_max_decay)
a_s = np.append(a_s, np.ones(T_max - T_max_decay))  # pad to full T_max length

# Weight schedule
all_weights_all = []
# First T_const iterations
for t in range(T_const):
    all_weights = []
    for n in range(N_partitions):
        weights = np.zeros(N)
        for i in range(N):
            if partition_assignment[i] == n:
                weights[i] = np.exp(a)
            else:
                weights[i] = np.exp(b)
        weights /= np.sum(weights)
        all_weights.append(weights)
    all_weights_all.append(all_weights)

# Remaining iterations with decay
for t in range(100):  # <-- still capped at 100 iterations here
    all_weights = []
    for n in range(N_partitions):
        weights = np.zeros(N)
        for i in range(N):
            if partition_assignment[i] == n:
                weights[i] = np.exp(a_s[t])
            else:
                weights[i] = np.exp(b)
        weights /= np.sum(weights)
        all_weights.append(weights)
    all_weights_all.append(all_weights)

# Save partition info
partition_info = {
    "partitions": partitions,
    "partition_assignment": partition_assignment,
    "all_weights": all_weights_all
}

pickle.dump(
    partition_info,
    open(
        f"aux_files/partitions_info_N_part_{N_partitions}_a_{a}_b_{b}_T_decay_{T_max_decay}_T_const_{T_const}_adaptive.pkl",
        "wb"
    )
)
