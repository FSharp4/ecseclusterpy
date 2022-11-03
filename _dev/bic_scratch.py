import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist

from clustering import uniclustering

data = np.array([[6, 10, 4], [7, 6, 5], [3, 10, 9], [5, 1, 3]])

odm = np.square(squareform(pdist(data)))
fdm = np.square(squareform(pdist(data.transpose())))


uniclustering.linkage(data, naive=True)
uniclustering.linkage(data.transpose(), naive=True)
Z = hierarchy.linkage(data, method="ward")
print("Debug POint")