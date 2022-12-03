import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy

from clustering import uniclustering
from data import generatedata

SMALL_SIZE = 16
MEDIUM_SIZE = 24
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
if __name__ == "__main__":
    n = 1000
    m = 20
    c = 10
    Z1 = None
    Z2 = None
    Z3 = None
    t = 30

    matches_scipy = np.ndarray([t], dtype=bool)
    optimization_valid = np.ndarray([t], dtype=bool)

    for ii in range(t):
        print(f"{ii}")
        data = generatedata.generate(n, m, random.randint(2, c), random.uniform(0.5, 2.0), True).features
        Z1 = hierarchy.linkage(data, method="ward")
        Z2 = uniclustering.ward_linkage(data, naive=True)
        Z3 = uniclustering.ward_linkage(data, naive=False)
        matches_scipy[ii] = np.allclose(Z1, Z3)
        optimization_valid[ii] = np.allclose(Z2, Z3)

    print(matches_scipy)
    print(optimization_valid)

    data = generatedata.generate(10, m, 3, 1.5, True).features
    Z1 = hierarchy.linkage(data, method="ward")
    Z2 = uniclustering.ward_linkage(data, naive=True)
    Z3 = uniclustering.ward_linkage(data, naive=False)

    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    plt.subplot(1, 3, 1)
    hierarchy.dendrogram(Z1)
    plt.title("Scipy Ward Method")
    plt.ylabel("Increase in net ESS")
    plt.xlabel("Point Index")

    plt.subplot(1, 3, 2)
    hierarchy.dendrogram(Z2)
    plt.title("Naive Ward Method")
    plt.ylabel("Increase in net ESS")
    plt.xlabel("Point Index")

    plt.subplot(1, 3, 3)
    hierarchy.dendrogram(Z3)
    plt.title("NNC Ward Method")
    plt.ylabel("Increase in net ESS")
    plt.xlabel("Point Index")

    plt.show()

