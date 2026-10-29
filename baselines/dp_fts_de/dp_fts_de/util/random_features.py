import numpy as np


def generate_random_features(dim, M, ls=0.01, v_kernel=1.0, obs_noise=1e-6):
    """
    Generates random Fourier features for a Gaussian kernel approximation.

    Args:
        dim (int): The dimension of the input space.
        M (int): The number of random features to generate.
        ls (float, optional): The length-scale of the Gaussian kernel. Defaults to 0.01.
        v_kernel (float, optional): The variance of the Gaussian kernel. Defaults to 1.0.
        obs_noise (float, optional): The observation noise. Defaults to 1e-6.

    Returns:
        dict: A dictionary containing the parameters and the generated random features 's' and 'b'.
    """
    # 1. Sample random frequencies 's' from a multivariate normal distribution
    # The covariance matrix is scaled by the inverse squared length-scale
    covariance_matrix = 1 / (ls**2) * np.identity(dim)
    s = np.random.multivariate_normal(np.zeros(dim), covariance_matrix, M)

    # 2. Sample random phase shifts 'b' from a uniform distribution
    b = np.random.uniform(0, 2 * np.pi, M)

    # 3. Assemble the features and parameters into a dictionary
    random_features = {
        "M": M,
        "dim": dim,
        "length_scale": ls,
        "s": s,
        "b": b,
        "obs_noise": obs_noise,
        "v_kernel": v_kernel
    }

    return random_features


# --- Example Usage ---
if __name__ == "__main__":
    # Define the input parameters
    input_dimension = 2
    num_features = 100

    # Generate the random features by calling the function
    features = generate_random_features(dim=input_dimension, M=num_features)

    # Print some results to verify the output
    print(f"--- Generated Random Features (dim={input_dimension}, M={num_features}) ---")

    print(f"Dictionary keys: {features.keys()}")

    # Check the shapes of the generated arrays
    print(f"Shape of frequency matrix 's': {features['s'].shape}")  # Expected: (100, 2)
    print(f"Shape of phase vector 'b': {features['b'].shape}")     # Expected: (100,)

    print("\nFirst 5 frequency vectors (s):")
    print(features['s'][:5])
