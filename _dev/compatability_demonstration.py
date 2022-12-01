from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram

from clustering import uniclustering
from data import generatedata
from information import dendrogram_cut

if __name__ == "__main__":
    data = generatedata.generate(n_samples=50, n_features=25, centers=5, cluster_std=2, shuffle=True, seed=25).features
    Z = uniclustering.ward_linkage(data)
    Z2 = hierarchy.linkage(data, method="ward")
    significant_clusters = dendrogram_cut.scheme3(data, Z)
    cut_d = dendrogram_cut.find_cutting_distance(Z, significant_clusters)

    print(significant_clusters)
    plt.figure()
    dendrogram(Z, no_labels=True, color_threshold=cut_d)
    # plt.axhline(linestyle='--', y=cut_d)
    plt.title("Custom Ward Clustering dendrogram")
    plt.ylabel("Sum of Cluster Variance")
    plt.xlabel("Cluster (by color)")
    plt.show()

    plt.figure()
    dendrogram(Z2, no_labels=True, color_threshold=cut_d)
    # plt.axhline(linestyle='--', y=cut_d)
    plt.title("Reference Ward Clustering dendrogram")
    plt.ylabel("Sum of Cluster Variance")
    plt.xlabel("Cluster (by color)")
    plt.show()
