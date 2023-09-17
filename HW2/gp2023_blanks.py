from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn, randint
from scipy.linalg import cholesky


# We will assume that mean is 0 for GP (but note that a conditional mean may not be 0.)

def safe_cholesky(K: np.ndarray) -> np.ndarray:
    # returns lower triangular matrix using Cholesky decomposition.
    return cholesky(K + 1e-6 * np.eye(len(K)), lower=True)


def get_SE_kernel(length_scale: float = 1,
                  magnitude: float = 1.) -> Callable[[np.ndarray], np.ndarray]:
    # returns a method that computes a kernel matrix given data (n x p matrix, but we will just use n x 1 matrix)
    # based on the given length scale (l) and magnitude (sigma**2)
    def SE_kernel(X: np.ndarray) -> np.ndarray:
        return squared_exponential(X, length_scale, magnitude)

    return SE_kernel


def squared_exponential(X: np.ndarray,
                        length_scale: float = 1.0,
                        magnitude: float = 1.0) -> np.ndarray:
    # TODO returns an SE kernel matrix of data X (n x p matrix, where we assume p = 1 for simplicity)
    # TODO Please check out the GP lecture slides page 17 and the working code in page 34,
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = X[i] - X[j]
            K[i, j] = magnitude ** 2 * np.exp(-0.5 * np.sum(diff ** 2) / length_scale ** 2)
    return K

def prior_sampling(X: np.ndarray,
                   K_of: Callable[[np.ndarray], np.ndarray],
                   num_samples: int = 1) -> np.ndarray:
    # TODO returns an n x num_samples matrix where each column represents sampled Y values w.r.t X following GP
    n = len(X)
    K = K_of(X)
    L = safe_cholesky(K)
    samples = L @ randn(n, num_samples)
    return samples


def posterior_sampling(test_X: np.ndarray,
                       K_of,
                       data_X: np.ndarray,
                       data_Y: np.ndarray,
                       num_samples: int = 1,
                       noise_variance: float = 0):
    # TODO returns an n x num_samples matrix
    # TODO where each column represents sampled Y values w.r.t. test_X following a GP given data points (data_X and data_Y)
    n1, n2 = len(test_X), len(data_X)
    X = np.vstack((test_X, data_X))

    # TODO compute a kernel matrix for X. Split it into 4 blocks and use 3 of them (A,B,C)
    # TODO ensure that a noise is added to C to address noisy observations.
    K = K_of(X)
    A, B, C = K[:n1, :n1], K[:n1:, n1:], K[n1:, n1:]
    C += noise_variance * np.eye(n2)
    C_inv = np.linalg.inv(C)
    # TODO compute mu and K conditional on the given data
    cond_mu = B @ np.linalg.solve(C, data_Y)
    cond_K = A - (B @ C_inv) @ B.T

    # TODO Use cholesky decomposition and random Gaussian noises to create multiple samples. Make sure to add cond_mu.
    L = safe_cholesky(cond_K)
    samples = L @ randn(n1, num_samples) + cond_mu
    return samples


def draw_GP(X, Ys, data_X=None, data_Y=None, fname=None):
    plt.figure(figsize=(7, 4))

    for Y in Ys.T:
        plt.plot(X, Y, 'b', alpha=0.25, linewidth=0.2, zorder=1)

    if data_X is not None and data_Y is not None:
        plt.scatter(data_X, data_Y, c='r', s=2, zorder=2)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()


def main():
    np.random.seed(0)

    plot_range = 40  # range of X [0, plot_range]
    num_functions = 150  # How many functions we will sample for prior and posterior.
    n = 200  # a number of data points to plot
    n_given = n // 10  # a number of instances in a given data for posterior sampling
    noise_sigma = 0.3  # We will assume a measurement error (noisy environment) for posterior sampling

    for length_scale, sigma in [(1, 1), (3, 1), (1, 3)]:
        kernel = get_SE_kernel(length_scale, sigma ** 2)

        # Prior sampling
        # We will have uniformly distributed X between 0 and `plot_range`
        X = np.arange(0, plot_range, plot_range / n).reshape((-1, 1))
        Ys = prior_sampling(X, kernel, num_functions)
        draw_GP(X, Ys, fname=f'prior_{length_scale}_{sigma}.pdf')

        # Posterior sampling
        # Assume that one of prior sample as a true one, let choose the first one (0:1)
        # As noisy data points, randomly select a few rows based on a prior sample.
        idxs = randint(n, size=n_given)
        given_X, given_Y = X[idxs], Ys[idxs, 0:1] + noise_sigma ** 2 * randn(n_given, 1)
        # Sample multiple functions based on given data.
        predicted_Ys = posterior_sampling(X, kernel, given_X, given_Y, num_functions,
                                          noise_variance=noise_sigma ** 2)
        draw_GP(X, predicted_Ys, given_X, given_Y, fname=f'posterior_{length_scale}_{sigma}.pdf')


if __name__ == '__main__':
    main()
