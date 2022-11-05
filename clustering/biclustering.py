from typing import List

import numpy as np

from clustering import uniclustering


def edist(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    m = data.shape[1]
    element_distance_vectors = np.ndarray([int((n * (n - 1)) / 2), m])
    ptr = 0
    for ii in range(n):
        for jj in range(ii + 1, n):
            element_distance_vectors[ptr] = np.square(data[ii] - data[jj])
            ptr += 1

    return element_distance_vectors


def cubeform(element_distance_vectors) -> np.ndarray:
    """
    Constructs a tensor with three numeric dimensions (two corresponding to the row dimension of the matrix and one to
    the column dimension) and one flag dimension (for signalling inclusion in minimization searches).
    :param element_distance_vectors: Column element-wise distances between rows in a matrix
    :return: The "cube" tensor
    """
    n = int((2 + np.sqrt(4 + 8 * element_distance_vectors.shape[0])) / 2)
    m = element_distance_vectors.shape[1]
    distance_tensor = np.zeros([n, n, m])
    ptr = 0
    for ii in range(n):
        distance_tensor[ii, ii, :] = np.nan
        for jj in range(ii + 1, n):
            distance_tensor[ii, jj, :] = element_distance_vectors[ptr]
            distance_tensor[jj, ii, :] = element_distance_vectors[ptr]
            ptr += 1

    return distance_tensor


def naive_separated(data: np.ndarray):
    Z_features = uniclustering.linkage(data)
    Z_columns = uniclustering.linkage(data.T)
    return Z_features, Z_columns


def naive_integrated_slow(data: np.ndarray):
    n = data.shape[0]
    m = data.shape[1]
    clustering_sizes = np.ones([n + m, 2], dtype=int)
    clustering_sizes[:n, 0] = np.arange(0, n)
    clustering_sizes[n:, 0] = np.arange(0, m)
    observation_distance_tensor = cubeform(edist(data))
    feature_distance_tensor = cubeform(edist(data.T))
    observation_linkage_matrix = np.zeros([n - 1, 4])
    feature_linkage_matrix = np.zeros([m - 1, 4])
    ordering_vector = np.ndarray([n + m - 2], dtype=bool)
    o_iteration = 0
    f_iteration = 0
    for iteration in range(n + m - 2):
        o_min_dist = tensor3_find_minimum(observation_distance_tensor)
        f_min_dist = tensor3_find_minimum(feature_distance_tensor)
        min_dist: List
        if o_min_dist[2] <= f_min_dist[2]:
            ordering_vector[iteration] = True
            min_dist = o_min_dist
            size_k = np.tile(clustering_sizes[:n, 1], (observation_distance_tensor.shape[2], 1)).T
        else:
            ordering_vector[iteration] = False
            min_dist = f_min_dist
            min_dist[0] += n
            min_dist[1] += n
            size_k = np.tile(clustering_sizes[n:, 1], (feature_distance_tensor.shape[2], 1)).T

        high_cluster_idx = min_dist[0]
        low_cluster_idx = min_dist[1]
        true_cluster_indexes = np.array(
            [clustering_sizes[high_cluster_idx, 0], clustering_sizes[low_cluster_idx, 0]], dtype=int
        )
        true_cluster_indexes.sort()
        size_i = clustering_sizes[low_cluster_idx, 1]
        size_j = clustering_sizes[high_cluster_idx, 1]
        merged_size = size_i + size_j
        linkage_vector = [true_cluster_indexes[0], true_cluster_indexes[1], np.sqrt(min_dist[2]), merged_size]
        if ordering_vector[iteration]:
            _run_update(feature_distance_tensor, high_cluster_idx, linkage_vector, low_cluster_idx, o_iteration,
                        observation_distance_tensor, observation_linkage_matrix, size_i, size_j, size_k)
            clustering_sizes[high_cluster_idx] = [-1, 0]
            clustering_sizes[low_cluster_idx] = [n + o_iteration, merged_size]
            o_iteration += 1
        else:
            _run_update(observation_distance_tensor, high_cluster_idx - n, linkage_vector, low_cluster_idx - n,
                        f_iteration, feature_distance_tensor, feature_linkage_matrix, size_i, size_j, size_k)
            clustering_sizes[high_cluster_idx] = [-1, 0]
            clustering_sizes[low_cluster_idx] = [m + f_iteration, merged_size]
            f_iteration += 1

    return observation_linkage_matrix, feature_linkage_matrix, ordering_vector


def _run_update(static_variance_tensor, high_cluster_idx, linkage_vector, low_cluster_idx, dim_iteration,
                modify_variance_tensor, linkage_matrix, size_i, size_j, size_k):
    linkage_matrix[dim_iteration] = linkage_vector
    distance_slice = modify_variance_tensor[low_cluster_idx, :, :]
    s_ijk = size_i + size_j + size_k
    distance_slice = (size_k + size_i) / s_ijk * distance_slice + (size_k + size_j) / s_ijk \
                     * modify_variance_tensor[high_cluster_idx, :, :] - size_k / s_ijk \
                     * distance_slice[high_cluster_idx, :]

    modify_variance_tensor[low_cluster_idx, :, :] = distance_slice
    modify_variance_tensor[:, low_cluster_idx, :] = distance_slice
    modify_variance_tensor[high_cluster_idx, :, :] = np.nan
    modify_variance_tensor[:, high_cluster_idx, :] = np.nan
    other_bicluster_distance_slice = static_variance_tensor[:, :, high_cluster_idx]
    other_bicluster_distance_slice[np.isnan(other_bicluster_distance_slice)] = 0
    static_variance_tensor[:, :, low_cluster_idx] += other_bicluster_distance_slice
    static_variance_tensor[:, :, high_cluster_idx] = np.nan


def tensor3_find_minimum(tensor3) -> List:
    min_dist = [-1, -1, np.inf]
    dimensions = tensor3.shape
    for ii in range(dimensions[0]):
        for jj in range(ii):
            if not np.isnan(tensor3[ii,jj]).all():
                candidate = np.nansum(tensor3[ii, jj])
                if candidate < min_dist[2]:
                    min_dist = [ii, jj, candidate]
                    if candidate == 0:
                        return min_dist

    return min_dist
