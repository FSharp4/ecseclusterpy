import random

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from data.io import writer


class Dataset:  # TODO: Investigate replacing this with pd.DataFrame
    features: any
    clusters: any
    headings: list

    def __init__(self, features, clusters):
        self.features = features
        self.clusters = clusters
        self.headings = feature_headings(features[0].__len__())


def generate_gan() -> Dataset:
    "This is from the Data Clustering textbook, Fig. 7.9"
    __prebaked_data__ = numpy.array(numpy.float64([[1.0, 2.0], [1.0, 2.5], [3.0, 1.0], [4.0, 0.5], [4.0, 2.0]]))
    prebaked_data = Dataset(__prebaked_data__, numpy.array(numpy.int64([1, 1, 1, 1, 1])))
    return prebaked_data


def prebaked_data2() -> numpy.ndarray:
    pd = numpy.array(numpy.float64([
        [4.115020313, 9.208512652, -6.935319543], [-5.915804542, 8.34092723, -2.049676389],
        [6.51548277, 6.719136511, -2.033675084], [-5.09667001, -3.13086089, 2.163730112],
        [2.134049462, 9.98065092, -8.022658998], [8.764541953, 7.216721423, 7.367159205],
        [5.281157828, 8.300911258, -7.829373185], [4.297429227, 4.09870539, -0.5793221953],
        [-2.661417389, -6.032691539, -4.149879742], [-3.122537551, 1.246373501, -5.248282474],
        [3.19707399, -0.6924246094, -5.566932629], [1.779631296, -4.085537816, -5.323861624],
        [3.497013586, 1.793871525, -1.889397862], [1.727592338, 8.107604018, 3.3142731],
        [5.259835313, -9.723197453, -4.496710247], [-4.776268907, -9.557114748, -3.55881477],
        [-1.978928732, -9.148571436, -2.628830179], [6.071187514, 6.66112668, -5.032432438],
        [0.9331776732, 5.94684077, -4.642818441], [-3.256444273, -9.70219793, -8.246416993],
        [8.283007056, -6.782928315, -2.601255383], [-3.837003458, 3.304297119, -1.571083685],
        [7.8125353, -3.5207574, -5.942607809], [-2.483136527, -0.08706439528, -1.792752014],
        [-3.947622442, 5.642470786, 4.375879296], [-5.456995013, 0.2686727492, -1.216256369],
        [7.048433913, 9.208427987, -2.676276341], [-0.1264137717, 6.873115872, -2.206216921],
        [-8.582205966, -4.025879092, -6.654093663], [-9.762020983, -5.668664566, 2.556566141]
    ]))
    return pd


def prebaked_data2_set() -> Dataset:
    pd = prebaked_data2()
    return Dataset(pd, [1] * len(pd))


def generate_example() -> Dataset:
    features, clusters = make_blobs(n_samples=2000, n_features=10, centers=5, cluster_std=0.4, shuffle=True)
    return Dataset(features, clusters)


def generate_random() -> Dataset:
    n_features = random.randint(2, 16)
    centers = random.randint(2, 16)
    cluster_std = random.uniform(0.1, 2.0)
    features, clusters = make_blobs(n_samples=2000, n_features=n_features, centers=centers, cluster_std=cluster_std,
                                    shuffle=True)
    return Dataset(features, clusters)


def generate(n_samples, n_features, centers, cluster_std, shuffle):
    """Wrapper function"""
    features, clusters = make_blobs(n_samples, n_features, centers=centers, cluster_std=cluster_std, shuffle=shuffle)
    return Dataset(features, clusters)


def generate_simple(n_samples, n_features):
    return generate(n_samples, n_features, centers=random.randint(2, int(np.sqrt(n_samples))),
                    cluster_std=random.random() * 1.5 + 0.5, shuffle=True)


def RAW_generate_example() -> numpy.ndarray:
    return generate_example().features


def feature_headings(n: int) -> list:
    headings = []
    for ii in range(1, n + 1):
        headings.append(f"Feature {ii}")

    return headings


def generate_romesburg() -> numpy.ndarray:
    return numpy.array([[10, 5], [20, 20], [30, 10], [30, 15], [5, 10]], dtype=int)


if __name__ == '__main__':
    data = generate_example()
    print("Feature matrix: ")
    print(pd.DataFrame(data.features, columns=data.headings))
    plt.scatter(data.features[:, 0], data.features[:, 1])
    plt.show()
    writer.write("example.csv", data)
    print("Done")
