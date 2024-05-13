import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_SPD_matrix(n):
    """
    Generate a random symmetric positive-semidefinite matrix of size n x n.

    Args:
    - n (int): Size of the square matrix.

    Returns:
    - A (array): Symmetric positive-semidefinite matrix of size n x n.
    """
    # Generate a random symmetric matrix
    A = np.random.randn(n, n)
    A = np.dot(A, A.T)

    # Ensure positive-semidefiniteness
    eigvals, eigvecs = np.linalg.eigh(A)
    A = np.dot(eigvecs, np.dot(np.diag(np.maximum(eigvals, 0)), eigvecs.T))
    A = A * np.random.uniform(0.00, 0.001)

    return A

def generate_nd_gaussian_mixture(n_samples, means, covariances, weights, dim):
    """
    Generate samples from an n-dimensional Gaussian mixture.

    Args:
    - n_samples (int): Number of samples to generate.
    - means (list of arrays): List of mean vectors for each component.
    - covariances (list of 2D arrays): List of covariance matrices for each component.
    - weights (list): List of weights for each component.
    - dim (int): Dimensionality of the data.

    Returns:
    - samples (array): Generated samples.
    """
    n_components = len(weights)
    samples_per_component = np.random.multinomial(n_samples, weights)

    samples = np.zeros((n_samples, dim))
    start = 0
    for i in range(n_components):
        n = samples_per_component[i]
        samples[start:start + n, :] = np.random.multivariate_normal(means[i], covariances[i], n)
        start += n

    return samples


# Define parameters for the mixture
n_samples = 10000
dim = 20  # Dimensionality of the data
n_components = 5  # Number of components

seed = 155
np.random.seed(seed)

# Generate random means, covariances, and weights for each component
means = [np.random.uniform(0, 1, dim) for _ in range(n_components)]
covariances = [generate_SPD_matrix(dim) for _ in range(n_components)]

weights = np.random.dirichlet(np.ones(n_components))

# Generate samples from the Gaussian mixture
samples = generate_nd_gaussian_mixture(n_samples, means, covariances, weights, dim)

# Save data
labels = []
for d in range(dim):
    labels.append('X'+str(d))
df = pd.DataFrame(samples)  
df.columns = labels
df.to_csv('mixture.csv', index=False)

# Plot the generated samples
plt.figure(figsize=(8, 6))
plt.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.5, markersize = 1.0)
plt.title('20D Gaussian Mixture')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
