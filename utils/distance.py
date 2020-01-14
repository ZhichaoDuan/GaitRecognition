import numpy as np

def compute_euclidean_dist(ndarr1, ndarr2):
    dist = np.expand_dims(np.sum(ndarr1 ** 2, axis=1), 1) + np.expand_dims(np.sum(ndarr2 ** 2, 1), 0) - 2 * np.matmul(ndarr1, np.transpose(ndarr2))
    dist = np.sqrt(np.maximum(0, dist))
    return dist

def compute_cosine_dist(ndarr1, ndarr2):
    dot = np.matmul(ndarr1, np.transpose(ndarr2))
    norm_arr1 = np.linalg.norm(ndarr1, axis=1).reshape(-1, 1)
    norm_arr2 = np.linalg.norm(ndarr2, axis=1).reshape(1, -1)
    norm_mat = np.matmul(norm_arr1, norm_arr2)
    return 1. - dot / norm_mat