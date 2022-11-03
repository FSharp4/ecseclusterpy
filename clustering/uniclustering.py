import math

import numpy
from scipy.spatial.distance import pdist, squareform

from data.vector_supplement import s_distance_squared


# import numpy as np


def _pdist(data):
    n = data.shape[0]
    distance_vector = numpy.ndarray([int((n * (n - 1)) / 2)])
    ptr = 0
    for ii in range(n):
        for jj in range(ii + 1, n):
            distance_vector[ptr] = s_distance_squared(data[ii], data[jj])
            ptr += 1

    return distance_vector


def _squareform(distance_vector) -> numpy.ndarray:
    n = math.floor((2 + math.sqrt(4 + 8 * distance_vector.size)) / 2)
    distance_matrix = numpy.zeros([n, n])
    ptr = 0
    for ii in range(n):
        for jj in range(ii + 1, n):
            distance_matrix[ii, jj] = distance_vector[ptr]
            distance_matrix[jj, ii] = distance_vector[ptr]
            ptr += 1

    distance_matrix[distance_matrix == 0] = math.nan
    return distance_matrix


def merge(acceptor: numpy.ndarray, donor: numpy.ndarray) -> None:
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
def positive_min(data: numpy.ndarray) -> tuple[float, int]:
    min_entry = float(numpy.nanmin(data))
    min_idx = int(numpy.argwhere(data == min_entry)[0][0])
    return min_entry, min_idx


def linkage(data: numpy.ndarray, naive: bool = False) -> numpy.ndarray:
    if naive:
        return _linkage_naive(data)

    n = data.shape[0]
    activity_size_matrix = numpy.ones([n, 2])
    activity_size_matrix[:, 0] = numpy.arange(0, n)
    """
    Format: original cluster id (for indexing in clusters), new cluster id (for linkage matrix), cardinality
    """
    distance_matrix: numpy.ndarray[numpy.ndarray[float]] = numpy.square(squareform(pdist(data)))
    distance_matrix[distance_matrix == 0] = math.nan
    linkage_matrix = numpy.ndarray([n - 1, 4])
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

    order = numpy.argsort(linkage_matrix[:, 2])
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
    return sorted_linkage_matrix


def _linkage_naive(data: numpy.ndarray) -> numpy.ndarray:
    n = len(data)
    activity_size_matrix = numpy.ones([n, 2])
    activity_size_matrix[:, 0] = numpy.arange(0, n)
    distance_matrix = numpy.square(squareform(pdist(data)))
    distance_matrix[distance_matrix == 0] = math.nan
    linkage_matrix = numpy.ndarray([n - 1, 4])
    for iteration in range(n - 1):
        min_D = numpy.nanmin(distance_matrix)
        min_coords = numpy.asarray(distance_matrix == min_D).nonzero()[0]
        if type(min_coords) == numpy.ndarray:
            min_coord = min_coords.reshape(int(min_coords.size / 2), 2)[0]
        else:
            min_coord = min_coords[0].reshape(min_coords.size / 2, 2)[0]

        min_coord.sort()
        low_cluster_idx = min_coord[0]
        high_cluster_idx = min_coord[1]
        true_cluster_idces = numpy.array(
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


def ess(data: numpy.ndarray, pairings: numpy.ndarray = None, series: bool = False) -> float | numpy.ndarray:
    """
    Exploratory usage, not towards linkage as this is inefficient
    :param pairings: Optional linkage pairings array (format row n = (cluster index i, cluster index j)
    :param data: Input data array
    :return: Either the ESS of the dataset, or the ESS of the dataset over every iteration of linkage
    """

    initial_ess = 0
    if pairings is None:
        mean = numpy.mean(data, axis=0)
        for i in range(data.shape[0]):
            diff = data[i] - mean
            initial_ess += numpy.dot(diff, diff)

        return initial_ess
    else:
        stored_clusters = numpy.arange(0, pairings.shape[0])
        for k in range(pairings.shape[0]):
            i = pairings[k, 0]
            j = pairings[k, 1]
            n = data.shape[0] + k - 1
            




