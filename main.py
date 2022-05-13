# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# read_from_existing = True
import time
import numpy

import generatedata
from ward import ward
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from pandas import read_csv

import numpy as np


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
    plt.show()
    times = [t1 - t0, t2 - t1]
    return sum(Z1[:, 2] - Z2[:, 2])

def huge():
    data = generatedata.generate(n_samples=500, n_features=3, centers=20, cluster_std=0.8, shuffle=True).features
    times = process(data, "Huge")
    return times

def big():
    data = generatedata.generate(n_samples=100, n_features=3, centers=15, cluster_std=0.8, shuffle=True).features
    times = process(data, "Big")
    return times


def medium(): # Size 150
    # iris = read_csv("input/iris.csv")
    # data = iris[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
    data = generatedata.generate(n_samples=50, n_features=3, centers=10, cluster_std=0.8, shuffle=True).features
    times = process(data, "Medium")
    return times


def little(): # Size 5
    # data = generatedata.prebaked_data().features
    data = generatedata.generate(n_samples=10, n_features=3, centers=5, cluster_std=0.8, shuffle=True).features
    times = process(data, "Little")
    return times


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    times_little = little()
    times_medium = medium()
    times_big = big()
    times_huge = huge()

    times_seaborn = [times_little[0], times_medium[0], times_big[0], times_huge[0]]
    times_custom = [times_little[1], times_medium[1], times_big[1], times_huge[1]]
    print(times_seaborn)
    print(times_custom)
    fig, ax = plt.subplots()
    x = numpy.arange(len(times_seaborn))
    for i in range(0,len(times_seaborn)):
        y = [times_seaborn[i], times_custom[i]]
        b = ax.bar(x + i * 0.75, y, 0.75, bottom=0.001)

    ax.set_xticks(x + 0.75 / 2, labels=["Little", "Medium", "Big"])
    ax.set_yscale('log')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Elapsed Time')
    plt.show()
    i_custom = [5, 10, 50, 100, 500, 1000]
    i_scipy = [5, 10, 50, 100, 500, 1000, 5000, 10000]
    i_custom = np.logspace(1, 3, num=20, endpoint=True, dtype=int)
    i_scipy = np.logspace(1, 4.5, num=30, endpoint=True, dtype=int)
    time_custom = np.float64(i_custom)
    time_scipy = np.float64(i_scipy)
    for i in range(0, len(i_custom)):
        time_custom[i] = trial_custom(i_custom[i])

    for i in range(0, len(i_scipy)):
        time_scipy[i] = trial_scipy(i_scipy[i])

    print(time_custom)
    print(time_scipy)
    # d = generatedata.prebaked_data().features
    # Z = ward(d)
    # fig, ax = plt.subplots()
    # hierarchy.dendrogram(Z, ax=ax)
    # ax.set_xlabel("Datapoint Index (within Dataset)")
    # ax.set_ylabel("Linkage Distance")
    # ax.set_title("Dendrogram of Five-Point Dataset")
    # plt.show()
    print("Done")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
