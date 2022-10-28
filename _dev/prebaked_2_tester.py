import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

from clustering import uniclustering
from data.generatedata import prebaked_data2

if __name__ == "__main__":
    data = prebaked_data2()
    Z = uniclustering.linkage(data)
    # Z2 = hierarchy.linkage(smallest, method="ward")
    fig = plt.figure()
    hierarchy.dendrogram(Z, orientation='top')
    fig.suptitle("Prebaked Data Dendrogram (Uniclustering)")
    plt.show()
    print(Z)