import random
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy

from clustering import uniclustering
from data import generatedata

if __name__ == "__main__":
    n = 1000
    M = np.arange(25, 1025, 25)
    scipy_times = np.ndarray(M.shape)
    naive_times = np.ndarray(M.shape)
    nnc_times = np.ndarray(M.shape)
    ctr = 0
    for m in M:
        print(f"{m} features")
        data = generatedata.generate(n, m, centers=10, cluster_std=2, shuffle=True).features

        t0 = time.time()
        hierarchy.linkage(data, method="ward")
        t1 = time.time()
        uniclustering.ward_linkage(data, naive=True)
        t2 = time.time()
        uniclustering.ward_linkage(data, naive=False)
        t3 = time.time()

        scipy_times[ctr] = t1 - t0
        naive_times[ctr] = t2 - t1
        nnc_times[ctr] = t3 - t2
        ctr += 1

    plt.figure()
    plt.semilogy(M, scipy_times, label="Scipy Runtimes")
    plt.semilogy(M, naive_times, label="Naive Runtimes")
    plt.semilogy(M, nnc_times, label="Nearest-Neighbor Chain Runtimes")
    plt.legend()
    plt.xlabel("Number of features in dataset")
    plt.ylabel("Log(Runtime in Seconds)")
    plt.title("Ward Algorithm Runtime (by feature count)")
    plt.show()
