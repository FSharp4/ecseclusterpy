from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from clustering import uniclustering
from data import generatedata
from information import dendrogram_cut

if __name__ == "__main__":
    data = generatedata.generate(n_samples=500, n_features=10, centers=8, cluster_std=2, shuffle=True, seed=25).features
    Z = uniclustering.ward_linkage(data)
    significant_clusters = dendrogram_cut.scheme3(data, Z)
    cut_d = dendrogram_cut.find_cutting_distance(Z, significant_clusters)

    print(significant_clusters)
    plt.figure()
    dendrogram(Z, no_labels=True, color_threshold=cut_d)
    plt.axhline(linestyle='--', y=cut_d)
    plt.title("Dendrogram of 500-point Dataset with BIC/ESS Automated Cut")
    plt.ylabel("Sum of Cluster Variance")
    plt.xlabel("Cluster (by color)")
    plt.show()
