from scipy.stats import norm

from clustering import uniclustering, variance
from data import generatedata
from information import criteria

"""
How biclustering ward combines features is probably 
pythagorean (a^2 + b^2 = c^2, with new feature = c)
"""

# For easy debug:
# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
from scipy.spatial.distance import pdist, squareform
# noinspection PyUnresolvedReferences
from scipy.cluster.hierarchy import linkage, dendrogram

from matplotlib import pyplot as plt

if __name__ == "__main__":

    param = np.array([500, 5, 16, 1])
    dataset = generatedata.generate(n_samples=param[0], n_features=param[1], centers=param[2], cluster_std=param[3], shuffle=True)
    data = dataset.features
    Z = uniclustering.ward_linkage(data)
    plt.figure()
    dendrogram(Z)
    plt.title(f"Dendrogram of ward linkage on generated data ({str(param)})")
    plt.show()
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(f"Scatterplot of first two dimensions of dataset ({str(param)})")
    plt.show()
    information = criteria.calculate_bic_vector(data, Z)
    cluster_map = uniclustering.get_cluster_datapoint_mapping(Z)
    ess = variance.ward_ess(Z[:, 2])
    merged_clusters__x_ = np.arange(0, data.shape[0] - 1)

    desired_iteration = np.argmin(information[information.shape[0] // 2:]) + information.shape[0] // 2
    plt.figure()
    plt.plot(np.arange(0, data.shape[0]-1), information)
    plt.title(f"Plot of BIC over iteration, CUT @ {desired_iteration} iter, ({str(param)})")
    plt.show()

    # Experiment 0

    plt.figure()
    plt.plot(merged_clusters__x_, ess)
    plt.title(f"Plot of ess over iteration ({str(param)})")
    plt.show()

    plt.figure()
    plt.plot(merged_clusters__x_[::-1], (-ess + ess[-1]) / ess[-1])
    plt.title(f"Plot of explained ess over number of clusters ({str(param)})")
    plt.show()

    # Experiment 1


    gradient = np.gradient(information)

    plt.figure()
    plt.plot(merged_clusters__x_[-20:], gradient[-20:])
    plt.title(f"Zoomed-in plot of d[BIC]/d[iteration] ({str(param)})")
    plt.show()

    plt.figure()
    plt.plot(merged_clusters__x_, gradient)
    plt.title(f"Plot of d[BIC]/d[iteration] ({str(param)})")
    plt.show()

    # Experiment 2

    plt.figure()
    plt.plot(merged_clusters__x_, Z[:, 2])
    plt.title(f"Plot of clustering error over iteration ({str(param)}")
    plt.show()

    plt.figure()
    plt.plot(merged_clusters__x_[-20:], Z[-20:, 2], marker='o')
    plt.title(f"Zoomed-in plot of clustering error over iteration ({str(param)})")
    plt.show()

    # Experiment 3

    plt.figure()
    plt.plot(Z[:, 2], gradient, marker='o')
    plt.title(f"Plot of d[BIC]/d[iteration] over total clustering error ({str(param)})")
    plt.show()

    # Experiment 4

    plt.figure()
    plt.plot(merged_clusters__x_, information / (merged_clusters__x_ + 1) * np.log(merged_clusters__x_ + 2))
    plt.title(f"Plot of BIC/(iteration log(iteration + 1) over iteration ({str(param)})")
    plt.show()

    # Experiment 5

    plt.figure()
    plt.plot(np.gradient(Z[:, 2]), gradient, marker='o')
    plt.title(f"Plot of d[BIC]/d[iteration] over d[total clustering error]/d[iteration] ({str(param)})")
    plt.show()

    x = np.gradient(Z[:, 2])
    y = gradient
    r = np.linalg.norm(np.vstack([x, y]), axis=0)
    mean_r = np.mean(r)
    std_r = np.std(r)
    r_z = (r - mean_r) / std_r
    probability_of_significance = norm.cdf(r_z)
    significant_clusters: int = 0
    for i in range(data.shape[0] - 1):
        p = probability_of_significance[-(i+1)]
        if p < 0.95:
            significant_clusters = i
            break

    print(f"Significant Clusters: {significant_clusters}")


    print("Debug Point")
    print("Debug Point")
    print("Debug Point")