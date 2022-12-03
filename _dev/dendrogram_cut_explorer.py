import random

import numpy as np
from scipy.cluster import hierarchy

from data import generatedata
from information import dendrogram_cut

if __name__ == "__main__":
    cluster_stdevs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    n_features = 20
    n_observations = 500
    clusters = 5
    bicess_significant_clusters = np.zeros([len(cluster_stdevs), 5, 5])
    aicess_significant_clusters = np.zeros([len(cluster_stdevs), 5, 5])
    elbow_significant_clusters = np.zeros([len(cluster_stdevs), 5, 5])
    iptr = 0
    for stdev in cluster_stdevs:
        for i in range(5):
            seed = random.randint(0, 100000)
            # print(f"Seed: {seed}")
            data = generatedata.generate(n_samples=n_observations, n_features=n_features, centers=clusters,
                                         cluster_std=stdev, shuffle=True, seed=seed).features
            partitions = np.split(data, 5)
            for k in range(len(partitions)):
                # print(f"({i}, {k})")
                _data = np.ndarray([0, data.shape[1]])
                for _k in range(len(partitions)):
                    if _k != k:
                        _data = np.vstack([_data, partitions[_k]])

                _Z = hierarchy.linkage(_data, method="ward")
                bicess_significant_clusters[iptr, i, k] = dendrogram_cut.bicess_cut(_data, _Z)
                aicess_significant_clusters[iptr, i, k] = dendrogram_cut.aicess_cut(_data, _Z)
                elbow_significant_clusters[iptr, i, k] = dendrogram_cut.elbow_cut(_data, _Z)


        mean_clusters_detected = np.mean(bicess_significant_clusters[iptr, :, :])
        stdev_clusters_detected = np.std(bicess_significant_clusters[iptr, :, :])
        print(f"for {stdev} bicess detects {mean_clusters_detected} plusminus {stdev_clusters_detected}")
        mean_clusters_detected = np.mean(aicess_significant_clusters[iptr, :, :])
        stdev_clusters_detected = np.std(aicess_significant_clusters[iptr, :, :])
        print(f"for {stdev} aicess detects {mean_clusters_detected} plusminus {stdev_clusters_detected}")
        mean_clusters_detected = np.mean(elbow_significant_clusters[iptr, :, :])
        stdev_clusters_detected = np.std(elbow_significant_clusters[iptr, :, :])
        print(f"for {stdev} elbow detects {mean_clusters_detected} plusminus {stdev_clusters_detected}")

        iptr += 1
