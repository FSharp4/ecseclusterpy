# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# read_from_existing = True
import time

import generatedata
# from ward import ward
from nnchain import ward
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy

# import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def trial_custom(size):
    print(f"Custom trial of size {size}")
    data = generatedata.generate(n_samples=size, n_features=3, centers=1, cluster_std=0.8, shuffle=True).features
    t0 = time.time()
    Z = ward(data)
    t1 = time.time()
    return t1 - t0


def trial_scipy(size):
    print(f"Scipy trial of size {size}")
    data = generatedata.generate(n_samples=size, n_features=3, centers=1, cluster_std=0.8, shuffle=True).features
    t0 = time.time()
    Z = hierarchy.linkage(data, method="ward")
    t1 = time.time()
    return t1 - t0


def process(data, name):
    print(f'Trial: {name}')
    t0 = time.time()
    Z1 = hierarchy.linkage(data, method="ward")
    t1 = time.time()
    Z2 = ward(data)
    t2 = time.time()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    hierarchy.dendrogram(Z1, ax=axes[0], orientation='top')
    hierarchy.dendrogram(Z2, ax=axes[1], orientation='top')
    axes[0].set_title('Scipy/Seaborn')
    axes[1].set_title('Custom')
    axes[0].set_xlabel("Cluster Index")
    axes[1].set_xlabel("Cluster Index")
    axes[0].set_ylabel("Euclidean Distance between Cluster Centers")
    axes[1].set_ylabel("Euclidean Distance between Cluster Centers")
    plt.show()
    return [t1 - t0, t2 - t1]
    # return sum(Z1[:, 2] - Z2[:, 2])


def huge():
    data = generatedata.generate(n_samples=500, n_features=3, centers=20, cluster_std=0.8, shuffle=True).features
    times = process(data, "Huge")
    return times


def big():
    data = generatedata.generate(n_samples=100, n_features=3, centers=15, cluster_std=0.8, shuffle=True).features
    times = process(data, "Big")
    return times


def medium():  # Size 150
    # iris = read_csv("input/iris.csv")
    # data = iris[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
    data = generatedata.generate(n_samples=50, n_features=3, centers=10, cluster_std=0.8, shuffle=True).features
    times = process(data, "Medium")
    return times


def little():  # Size 5
    # data = generatedata.prebaked_data().features
    data = generatedata.generate(n_samples=10, n_features=3, centers=5, cluster_std=0.8, shuffle=True).features
    times = process(data, "Little")
    return times


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import numpy as np
    i_scipy = np.logspace(1, 3, num=30, endpoint=True, dtype=int)
    time_custom = np.float64(i_scipy)
    time_scipy = np.float64(i_scipy)
    for i in range(0, len(i_scipy)):
        time_custom[i] = trial_custom(i_scipy[i])

    for i in range(0, len(i_scipy)):
        time_scipy[i] = trial_scipy(i_scipy[i])

    for i in range(len(i_scipy)):
        print(f"{i_scipy[i]} \t {time_custom[i]} \t {time_scipy[i]}")

    little()
    print("Done")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
