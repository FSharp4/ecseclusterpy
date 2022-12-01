import seaborn as sns
from matplotlib import pyplot as plt

from clustering import biclustering
from data import generatedata

if __name__ == "__main__": # 50
    # data = generatedata.generate_gan().features
    data = generatedata.generate(n_samples=25, n_features=25, centers=4, cluster_std=1.5, shuffle=True, seed=10).features
    Z_r, Z_c = biclustering.reference_optimized(data)
    plt.figure()
    sns.clustermap(data, row_linkage=Z_r, col_linkage=Z_c, xticklabels=False, yticklabels=False)
    plt.ylabel("Feature Value Colormap")
    plt.show()
    print("Debug point")
