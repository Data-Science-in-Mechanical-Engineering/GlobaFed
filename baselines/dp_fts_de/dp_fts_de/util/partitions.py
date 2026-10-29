import numpy as np
import itertools


def generate_partition_info(d, Domain, N=10, N_partitions_single_dim=2, a=16, b=1, T_max=200, T_const=10, T_max_decay=30):
    """
    Creates partition information for distributed exploration.

    Args:
        d (int): The input dimension of the domain space.
        Domain (np.ndarray): A 1D array of shape (2,) representing the [min, max] of the domain.
        N (int, optional): The total number of agents. Defaults to 10.
        N_partitions_single_dim (int, optional): The number of regions to partition each dimension into. Defaults to 2.
        a (float, optional): The high weight parameter for adaptive weights. Defaults to 16.
        b (float, optional): The low weight parameter for adaptive weights. Defaults to 1.
        T_max (int, optional): The maximum number of iterations for weight schedules. Defaults to 200.
        T_const (int, optional): The number of initial iterations where 'a' remains constant. Defaults to 10.
        T_max_decay (int, optional): The number of iterations over which 'a' decays to 'b'. Defaults to 30.

    Returns:
        dict: A dictionary containing 'partitions', 'partition_assignment', and 'all_weights'.
    """
    Domain_width = Domain[1] - Domain[0]

    # --- 1. Generate the partitions ---
    partitions = []
    # Generate all combinations of partition indices for d dimensions
    indices_product = itertools.product(range(N_partitions_single_dim), repeat=d)

    for indices in indices_product:
        current_partition = []
        # For each dimension, calculate the lower and upper bounds based on its index
        for i in range(d):
            idx = indices[i]
            lower_bound = Domain[0] + idx * Domain_width / N_partitions_single_dim
            upper_bound = Domain[0] + (idx + 1) * Domain_width / N_partitions_single_dim
            current_partition.append([lower_bound, upper_bound])
        partitions.append(np.array(current_partition))

    # --- 2. Assign partitions to agents ---
    N_partitions = len(partitions)
    partition_assignment = np.arange(N) % N_partitions

    # --- 3. Calculate adaptive weights over time ---
    a_s = np.linspace(a, b, T_max_decay)
    # Ensure weight schedule is not longer than T_max
    if T_const + len(a_s) > T_max:
        remaining_len = T_max - T_const
        a_s = np.linspace(a, b, remaining_len) if remaining_len > 0 else np.array([])

    a_schedule = np.concatenate([
        np.full(T_const, a),
        a_s,
        np.full(max(0, T_max - T_const - len(a_s)), b)  # Fill the rest with b
    ])

    all_weights_all = []
    for current_a in a_schedule:
        all_weights_single_t = []
        for n in range(N_partitions):
            weights = np.zeros(N)
            for i in range(N):
                if partition_assignment[i] == n:
                    weights[i] = np.exp(current_a)
                else:
                    weights[i] = np.exp(b)
            weights = weights / np.sum(weights)
            all_weights_single_t.append(weights)
        all_weights_all.append(all_weights_single_t)

    # --- 4. Assemble the final dictionary ---
    partition_info = {
        "partitions": partitions,
        "partition_assignment": partition_assignment,
        "all_weights": all_weights_all
    }

    return partition_info


def generate_bounds_based_partition_info(d, bounds_list, N=10, a=16, b=1, T_max=200, T_const=10, T_max_decay=30):
    """
    Creates partition information for distributed exploration from a list of bounds.

    Each bound in the input list is projected across all 'd' dimensions to form a
    hyper-rectangular partition.

    Args:
        d (int): The input dimension of the domain space.
        bounds_list (list): A list where each element is a two-item list or tuple
                            representing the [lower_bound, upper_bound] for a partition.
        N (int, optional): The total number of agents. Defaults to 10.
        a (float, optional): The high weight parameter for adaptive weights. Defaults to 16.
        b (float, optional): The low weight parameter for adaptive weights. Defaults to 1.
        T_max (int, optional): The maximum number of iterations for weight schedules. Defaults to 200.
        T_const (int, optional): The number of initial iterations where 'a' remains constant. Defaults to 10.
        T_max_decay (int, optional): The number of iterations over which 'a' decays to 'b'. Defaults to 30.

    Returns:
        dict: A dictionary containing 'partitions', 'partition_assignment', and 'all_weights'.
    """
    # --- 1. Generate partitions from the input bounds list ---
    partitions = []
    for bound in bounds_list:
        # Project the single bound across all 'd' dimensions
        # e.g., for d=3 and bound=[0, 1], partition becomes [[0,1], [0,1], [0,1]]
        current_partition = np.tile(bound, (d, 1))
        partitions.append(current_partition)

    # --- 2. Assign partitions to agents ---
    N_partitions = len(partitions)
    if N_partitions == 0:
        raise ValueError("The 'bounds_list' cannot be empty.")
    
    # Assign agents to partitions in a round-robin fashion
    partition_assignment = np.arange(N) % N_partitions

    # --- 3. Calculate adaptive weights over time (logic is unchanged) ---
    a_s = np.linspace(a, b, T_max_decay)
    # Ensure weight schedule does not exceed T_max
    if T_const + len(a_s) > T_max:
        remaining_len = T_max - T_const
        a_s = np.linspace(a, b, remaining_len) if remaining_len > 0 else np.array([])

    a_schedule = np.concatenate([
        np.full(T_const, a),
        a_s,
        np.full(max(0, T_max - T_const - len(a_s)), b)  # Fill the rest with 'b'
    ])

    all_weights_all = []
    for current_a in a_schedule:
        all_weights_single_t = []
        for n in range(N_partitions):
            weights = np.zeros(N)
            # Assign high weight to agents in the current partition, low weight to others
            assigned_agents_mask = (partition_assignment == n)
            weights[assigned_agents_mask] = np.exp(current_a)
            weights[~assigned_agents_mask] = np.exp(b)
            
            # Normalize the weights
            weights /= np.sum(weights)
            all_weights_single_t.append(weights)
        all_weights_all.append(all_weights_single_t)

    # --- 4. Assemble the final dictionary ---
    partition_info = {
        "partitions": partitions,
        "partition_assignment": partition_assignment,
        "all_weights": all_weights_all
    }

    return partition_info


# --- Example Usage ---
if __name__ == "__main__":
    # Define the input parameters
    input_dim = 3
    input_domain = np.array([-5, 5])

    # Generate the partition information by calling the function
    info = generate_partition_info(
        d=input_dim, Domain=input_domain, N=10, N_partitions_single_dim=2)

    # Print some results to verify the output
    print(f"--- Results for d={input_dim}, Domain={input_domain} ---")

    num_partitions = len(info['partitions'])
    print(f"Total number of partitions: {num_partitions}")

    print(f"Shape of the first partition: {info['partitions'][0].shape}")
    print("First partition bounds:")
    print(info['partitions'][0])

    print("\nPartition assignments for 20 agents:")
    print(info['partition_assignment'])

    print(f"\nTotal number of weight timesteps generated: {len(info['all_weights'])}")
    print(f"Shape of weights for t=0: {np.array(info['all_weights'][0]).shape}")

    print(info['partitions'])
