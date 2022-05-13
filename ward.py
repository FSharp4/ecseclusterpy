# 1. Calculate the mean location of each cluster
# 2. Calculate the ESS between each cluster
# 3. Calculate the Ward Linkage Function between each cluster using the ESS
from numpy import ndarray
from math import sqrt
from Cluster import Cluster, merge
from generatedata import Dataset


def ward(data: Dataset) -> ndarray:
    """
    Generates the linkage information using the Ward algorithm and a Euclidean distance metric.

    :param data: Set of data points formatted as a dataset.
    :return: An array describing the linkages (see scipy.cluster.hierarchy.linkage for format).
    """
    # Init
    n = data.__len__()
    clusters = [None] * n
    # for datapoint in data:
    for i in range(0, data.__len__()):
        datapoint = data[i]
        asArray = ndarray([1, datapoint.__len__()])
        asArray[0] = [dim for dim in datapoint]
        cluster = Cluster(datapoint)
        cluster.index = i
        clusters[i] = cluster

    linkageOutput = ndarray([n - 1, 4])  # This array matches format of scipy linkage method output
    """
    More specifically, scipy returns an n-1 x 4 array Z where:
    - Z(i, 0) and Z(i, 1) correspond to the indices of the merged clusters
    - Z(i, 2) corresponds to the distance between the merged clusters
    - Z(i, 3) corresponds to the number of observations in the merged cluster
    """

    D = ndarray([n, n])
    running_D = 0

    # Construct initial deltaESS and dissimilarity values
    for i in range(0, n):
        for j in range(i, n):
            if i != j:
                D[i][j] = clusters[i].distance_to(clusters[j].mean) ** 2
            else:
                D[i][j] = -1  # having D == 0 is indicative of invalid pair

            D[j][i] = D[i][j]  # Take advantage of symmetry

    for iteration in range(0, n - 1):
        i_min = 0
        j_min = 0
        min_D = float('inf')
        # Investigate every cluster pair for minimized ESS
        for i in range(0, n):
            for j in range(i + 1, n):
                if min_D > D[i][j] >= 0:  # negative D signals exclusion from consideration
                    min_D = D[i][j]
                    i_min = i
                    j_min = j

        # Make the merged cluster, and prepare to overwrite cluster i information
        newCluster = merge(clusters[i_min], clusters[j_min])
        newCluster.index = n + iteration
        running_D += D[i_min][j_min]
        lowCluster = min(clusters[i_min].index, clusters[j_min].index)
        highCluster = max(clusters[i_min].index, clusters[j_min].index)
        linkageOutput[iteration] = [
            lowCluster,
            highCluster,
            sqrt(D[i_min][j_min]),
            clusters[i_min].size + clusters[j_min].size
        ]
        for k in range(0, n):
            if k == i_min or k == j_min or D[i_min][k] == -1:
                continue

            sum_ijk = clusters[i_min].size + clusters[j_min].size + clusters[k].size
            # Overwrite cluster i information
            D[k][i_min] = (clusters[k].size + clusters[i_min].size) / sum_ijk * D[i_min][k] + \
                (clusters[k].size + clusters[j_min].size) / sum_ijk * D[k][j_min] - \
                clusters[k].size / sum_ijk * D[i_min][j_min]
            D[i_min][k] = D[k][i_min]

            # Destroy J cluster information (avoid deletion to prevent array copies)
            D[j_min][k] = -1
            D[k][j_min] = -1

        D[i_min][j_min] = -1
        D[j_min][i_min] = -1
        clusters[i_min] = newCluster  # Necessary for accurate D/ESS calculations after iteration 1

    return linkageOutput
