import numpy as np
from scipy.spatial.distance import squareform, pdist


# def linkage(data: np.ndarray) -> np.ndarray:
#     n_observations = data.shape[0]
#     n_features = data.shape[1]
#     observation_cluster_size_matrix = np.zeros([n_observations, 2])
#     feature_cluster_size_matrix = np.zeros([n_observations, 2])
#     observation_cluster_size_matrix[:, 0] = np.arange(0, n_observations)
#     feature_cluster_size_matrix[:, 0] = np.arange(0, n_features, -1)
#     observation_distance_matrix = squareform(pdist(data))
#     feature_distance_matrix = squareform(pdist(data.transpose()))
#     n_iterations = n_observations + n_features - 2
#     linkage_matrix = np.ndarray([n_iterations, 4])
#     stack_observation_cluster_indexes = []
#     stack_feature_clsuter_indexes = []
#     for iteration in n_iterations:
#         while True:

def naive(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    m = data.shape[1]
    activity_size_matrix = np.ones([n + m, 2])
    activity_size_matrix[:n, 0] = np.arange(0, n)
    activity_size_matrix[n:, 0] = np.arange(0, m)
    observation_distance_matrix = np.square(squareform(pdist(data)))
    feature_distance_matrix = np.