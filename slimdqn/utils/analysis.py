import numpy as np


def compute_srank(feature_matrix, delta=0.01):
    singular_vals = np.linalg.svd(feature_matrix, full_matrices=False, compute_uv=False)
    sorted_singular_vals = np.sort(singular_vals)[::-1]
    singular_vals_cumsum = np.cumsum(sorted_singular_vals)
    return np.searchsorted(singular_vals_cumsum, (1 - delta) * singular_vals_cumsum[-1], side="left") + 1


def compute_dead_neurons(score_neurons, tau=0):
    dead_neurons = 0
    total_neurons = 0
    for score in score_neurons:
        dead_neurons += np.count_nonzero(score / (score.mean() + 1e-9) <= tau)
        total_neurons += score.size
    return dead_neurons / total_neurons
