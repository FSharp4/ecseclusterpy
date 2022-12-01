import math
from typing import List

import numpy as np
from scipy.spatial.distance import pdist, squareform

from data.vector_supplement import s_distance_squared
from information import criteria


# import np as np


def _pdist(data):
    n = data.shape[0]
    distance_vector = np.ndarray([int((n * (n - 1)) / 2)])
    ptr = 0
    for ii in range(n):
        for jj in range(ii + 1, n):
            distance_vector[ptr] = s_distance_squared(data[ii], data[jj])
            ptr += 1

    return distance_vector


def _squareform(distance_vector) -> np.ndarray:
    n = math.floor((2 + math.sqrt(4 + 8 * distance_vector.size)) / 2)
    distance_matrix = np.zeros([n, n])
    ptr = 0
    for ii in range(n):
        for jj in range(ii + 1, n):
            distance_matrix[ii, jj] = distance_vector[ptr]
            distance_matrix[jj, ii] = distance_vector[ptr]
            ptr += 1

    distance_matrix[distance_matrix == 0] = math.nan
    return distance_matrix


def merge(acceptor: np.ndarray, donor: np.ndarray) -> None:
    """
    Merges donor into acceptor.
    Both acceptor and c2 are modified in-place:
      acceptor <= (acceptor + donor) / 2
      donor <= [NaN] * n
    :param acceptor: 1D array, modified in place
    :param donor: 1D array, of equal length.
    :return: Nothing; arrays are modified in-place
    """
    for ii in range(0, acceptor.size):
        acceptor[ii] = (acceptor[ii] + donor[ii]) / 2

    donor[:] = math.nan


# Linear
def positive_min(data: np.ndarray) -> tuple[float, int]:
    min_entry = float(np.nanmin(data))
    min_idx = int(np.argwhere(data == min_entry)[0][0])
    return min_entry, min_idx


# def ward_linkage_with_cut(data: np.ndarray, naive: bool = False, use_ess: bool = False,
#                           criterion=bayesian_criterion.bic):
#     Z = ward_linkage(data, naive, use_ess)
#     ess = np.square(0.5 * Z[:, 2])


def ward_linkage(data: np.ndarray, naive: bool = False, use_ess: bool = False) -> np.ndarray:
    if naive:
        return linkage_naive(data)

    n = data.shape[0]
    activity_size_matrix = np.ones([n, 2])
    activity_size_matrix[:, 0] = np.arange(0, n)
    """
    Format: original cluster id (for indexing in clusters), new cluster id (for linkage matrix), cardinality
    """
    distance_matrix: np.ndarray[np.ndarray[float]] = np.square(squareform(pdist(data)))
    distance_matrix[distance_matrix == 0] = math.nan
    linkage_matrix = np.ndarray([n - 1, 4])
    stack_clusters_indexes = []
    """
    `stack_clusters_indexes` stores the corresponding index of the cluster in `clusters` 
    within a stack such that
    ```
    stack[a] index == stack_clusters_indexes[a]
    ```
    """

    clusters_ptr = 0
    iteration = 0
    while iteration < n - 1:
        # 1. If stack is empty, push an arbitrary cluster onto the stack.
        #  - This should be the recently merged cluster if the stack is empty.
        if stack_clusters_indexes.__len__() == 0:
            stack_clusters_indexes.append(clusters_ptr)

        # 2. Examine the top cluster of the stack.
        top_cluster_idx = stack_clusters_indexes[-1]

        # 3. Calculate the distances between the top cluster and all other clusters
        distance_vector = distance_matrix[top_cluster_idx, :]
        # Select the cluster which is closest
        min_dist, min_cluster_idx = positive_min(distance_vector)
        ess = 0

        """
        4. If cluster is in stack, pop both clusters from stack and merge.
           Else, push cluster onto stack.
        """
        if min_cluster_idx == stack_clusters_indexes[len(stack_clusters_indexes) - 2]:
            """
            The above condition was a stumper. It previously read:
            > if min_cluster_idx in stack_clusters_indexes

            From wikipedia:
            > "If D is already in S, it must be the immediate predecessor of C."

            This has proven to be false in practice, probably due to the in-place storage of merged clusters over old 
            clusters that were in the stack, or some other mismatch between that concept and this implementation.
            """
            # Record output in linkage_matrix
            merged_size = activity_size_matrix[min_cluster_idx][1] + activity_size_matrix[top_cluster_idx][1]
            linkage_matrix[iteration] = [
                min(activity_size_matrix[min_cluster_idx][0], activity_size_matrix[top_cluster_idx][0]),
                max(activity_size_matrix[min_cluster_idx][0], activity_size_matrix[top_cluster_idx][0]),
                math.sqrt(min_dist),
                merged_size
            ]
            clusters_ptr = top_cluster_idx

            # Merge elements
            distance_vector = distance_matrix[top_cluster_idx, :]
            size_k = activity_size_matrix[:, 1]
            size_i = activity_size_matrix[top_cluster_idx, 1]
            size_j = activity_size_matrix[min_cluster_idx, 1]
            s_ijk = merged_size + size_k
            distance_vector = (size_k + size_i) / s_ijk * distance_vector + (size_k + size_j) / s_ijk * \
                              distance_matrix[min_cluster_idx, :] - size_k / s_ijk * \
                              distance_vector[min_cluster_idx]

            # Update distance matrix
            distance_matrix[min_cluster_idx, :] = math.nan
            distance_matrix[:, min_cluster_idx] = math.nan
            distance_matrix[top_cluster_idx, :] = distance_vector
            distance_matrix[:, top_cluster_idx] = distance_vector

            activity_size_matrix[min_cluster_idx] = [-1, 0]
            activity_size_matrix[top_cluster_idx] = [n + iteration, merged_size]
            iteration += 1

            # Remove both clusters from stack
            """
            These del lines were also changed with the condition adjustment. Maybe the culprit was here?
            """
            del stack_clusters_indexes[-1]
            del stack_clusters_indexes[-1]
        else:
            stack_clusters_indexes.append(min_cluster_idx)

    order = np.argsort(linkage_matrix[:, 2])
    sorted_linkage_matrix = linkage_matrix[order]
    for i in range(n - 1):
        entry = order[i]
        new_index = n + i
        old_index = n + entry
        if new_index == old_index:
            continue

        for j in range(n):
            if linkage_matrix[order[j]][0] == old_index:
                sorted_linkage_matrix[j][0] = new_index
                break
            elif linkage_matrix[order[j]][1] == old_index:
                sorted_linkage_matrix[j][1] = new_index
                break

    for j in range(n - 1):
        if sorted_linkage_matrix[j][0] > sorted_linkage_matrix[j][1]:
            sorted_linkage_matrix[j][0], sorted_linkage_matrix[j][1] = sorted_linkage_matrix[j][1], \
                                                                       sorted_linkage_matrix[j][0]

    if use_ess:
        sorted_linkage_matrix[:, 2] = np.cumsum(0.5 * np.square(sorted_linkage_matrix[:, 2]))

    return sorted_linkage_matrix


def linkage_naive(data: np.ndarray) -> np.ndarray:
    n = len(data)
    activity_size_matrix = np.ones([n, 2])
    activity_size_matrix[:, 0] = np.arange(0, n)
    distance_matrix = np.square(squareform(pdist(data)))
    distance_matrix[distance_matrix == 0] = math.nan
    linkage_matrix = np.ndarray([n - 1, 4])
    for iteration in range(n - 1):
        min_D = np.nanmin(distance_matrix)
        min_coords = np.asarray(distance_matrix == min_D).nonzero()[0]
        if type(min_coords) == np.ndarray:
            min_coord = min_coords.reshape(int(min_coords.size / 2), 2)[0]
        else:
            min_coord = min_coords[0].reshape(min_coords.size / 2, 2)[0]

        min_coord.sort()
        low_cluster_idx = min_coord[0]
        high_cluster_idx = min_coord[1]
        true_cluster_idces = np.array(
            [activity_size_matrix[low_cluster_idx][0], activity_size_matrix[high_cluster_idx][0]], dtype=int
        )
        true_cluster_idces.sort()
        size_i = activity_size_matrix[low_cluster_idx, 1]
        size_j = activity_size_matrix[high_cluster_idx, 1]
        merged_size = size_i + size_j
        linkage_matrix[iteration] = [true_cluster_idces[0], true_cluster_idces[1], math.sqrt(min_D), merged_size]
        distance_vector = distance_matrix[low_cluster_idx, :]
        size_k = activity_size_matrix[:, 1]
        s_ijk = merged_size + size_k
        distance_vector = (size_k + size_i) / s_ijk * distance_vector + (size_k + size_j) / s_ijk \
                          * distance_matrix[high_cluster_idx, :] - size_k / s_ijk * distance_vector[high_cluster_idx]

        distance_matrix[high_cluster_idx, :] = math.nan
        distance_matrix[:, high_cluster_idx] = math.nan
        distance_matrix[low_cluster_idx, :] = distance_vector
        distance_matrix[:, low_cluster_idx] = distance_vector

        activity_size_matrix[high_cluster_idx] = [-1, 0]
        activity_size_matrix[low_cluster_idx] = [n + iteration, merged_size]
        iteration += 1

    return linkage_matrix


"""
NOTE ON LINKAGE METHODS: 
These methods communicate the Ward variance criterion as distances in the 
linkage matrix, coded as the euclidean distance between the merged 
clusters.

Book methods (such as in G. Gan, or H. Romesburg use the error square sum 
(ESS), which can be obtained from the linkage matrix at each step by 
squaring the distance vector (linkage_matrix[:, 2]), dividing by two, and at 
each element i summating all elements with indices less than i.

For example, the linkage matrix obtained from generate_gan() is

[[0.        , 1.        , 0.5       , 2.        ],
[2.        , 3.        , 1.11803399, 2.        ],
[4.        , 6.        , 1.55456318, 3.        ],
[5.        , 7.        , 4.45907315, 5.        ]]

The minimum squared distance vector is 

[ 0.25      ,  1.25      ,  2.41666667, 19.88333333].

The ESS delta vector is 

[0.125     , 0.625     , 1.20833333, 9.94166667]

The ESS vector (obtainable from np.add.accumulate(ESS)) is
[ 0.125     ,  0.75      ,  1.95833333, 11.9       ]



"""


def increment_point_membership(cluster_membership, p1, p2, n) -> None:
    working_labels = cluster_membership[:, 1]
    working_labels[working_labels == p1] = n
    working_labels[working_labels == p2] = n


def calculate_point_membership(data, Z, cut_level: int):
    cluster_membership = np.vstack([np.arange(0, data.shape[0])] * 2).T
    working_labels = cluster_membership[:, 1]
    n = data.shape[0]
    for merge_pair in range(cut_level):
        p1, p2 = Z[merge_pair, :2]
        # working_labels[working_labels == p1] = n
        # working_labels[working_labels == p2] = n
        increment_point_membership(cluster_membership, p1, p2, n)
        n += 1

    return cluster_membership


def get_cluster_datapoint_mapping(Z: np.ndarray) -> dict[int, np.ndarray]:
    n_merged_nodes = Z.shape[0]
    n = n_merged_nodes + 1
    node_merge_LUT = np.hstack([np.atleast_2d(np.arange(0, n_merged_nodes)).T + n, Z[:, :2]]).astype(int)
    cluster_point_map: dict[int, np.ndarray] = {}

    for i in range(n):
        cluster_point_map[i] = np.array([i])

    for (i, p1, p2) in node_merge_LUT:
        merged1 = cluster_point_map[p1]
        merged2 = cluster_point_map[p2]
        cluster_point_map[i] = np.hstack([merged1, merged2])

    return cluster_point_map
