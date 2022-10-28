import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

from clustering import uniclustering
from data.generatedata import prebaked_data

if __name__ == "__main__":
    smallest = prebaked_data().features
    Z = uniclustering.linkage(smallest)
    # Z2 = hierarchy.linkage(smallest, method="ward")
    fig = plt.figure()
    hierarchy.dendrogram(Z, orientation='top')
    fig.suptitle("Prebaked Data Dendrogram (Uniclustering)")
    plt.show()
