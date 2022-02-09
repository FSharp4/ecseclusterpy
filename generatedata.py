import random

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs

import loader


class Dataset:
    features: any
    clusters: any

    def __init__(self, features, clusters):
        self.features = features
        self.clusters = clusters


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


def feature_headings(n: int) -> list:
    headings = []
    for ii in range(1, n + 1):
        headings.append(f"Feature {ii}")

    return headings


if __name__ == '__main__':
    data = generate_example()
    print("Feature matrix: ")
    print(pd.DataFrame(data.features,
                       columns=[feature_headings(data.features[0].__len__())]))
    plt.scatter(data.features[:, 0], data.features[:, 1])
    plt.show()
    loader.write("example.csv", data)
    print("Done")
