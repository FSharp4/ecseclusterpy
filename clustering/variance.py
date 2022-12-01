import numpy as np

from clustering import uniclustering
from clustering.uniclustering import increment_point_membership


def variance(dataset: np.ndarray) -> float:
    var = 0
    mean = np.mean(dataset, axis=0)
    for datapoint in dataset:
        residual = datapoint - mean
        var += np.sum(np.square(residual))

    return var / dataset.shape[0]

def get_clustering_variances(dataset, Z, n_clusters):
    n = dataset.shape[0]
    membership_matrix = uniclustering.calculate_point_membership(dataset, Z, n - 4 - 1)
    clusters = np.unique(membership_matrix[:, 1])
    variances = np.zeros([clusters.shape[0]])
    cluster_datapoint_mapping = uniclustering.get_cluster_datapoint_mapping(Z)
    for i in range(clusters.shape[0]):
        cluster_idx = clusters[i]
        points = cluster_datapoint_mapping[cluster_idx]
        datapoints = dataset[points]
        variances[i] = variance(datapoints)

    return variances

def ward_ess(ward_distance):
    return np.cumsum(0.5 * np.square(ward_distance))

