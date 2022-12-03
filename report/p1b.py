import random
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy

from clustering import uniclustering
from data import generatedata

if __name__ == "__main__":
    N = np.arange(50, 2000, 50)
    m = 30
    scipy_times = np.ndarray(N.shape)
    naive_times = np.ndarray(N.shape)
    nnc_times = np.ndarray(N.shape)
    ctr = 0
    for n in N:
        print(f"{n} datapoints")
        data = generatedata.generate(n, m, centers=10, cluster_std=2, shuffle=True).features

        t0 = time.time_ns()
        hierarchy.linkage(data, method="ward")
        t1 = time.time_ns()
        uniclustering.ward_linkage(data, naive=True)
        t2 = time.time_ns()
        uniclustering.ward_linkage(data, naive=False)
        t3 = time.time_ns()

        scipy_times[ctr] = t1 - t0
        naive_times[ctr] = t2 - t1
        nnc_times[ctr] = t3 - t2
        ctr += 1

    plt.figure()
    plt.semilogy(N, scipy_times * 1e-9, label="Scipy Runtimes")
    plt.semilogy(N, naive_times * 1e-9, label="Naive Runtimes")
    plt.semilogy(N, nnc_times * 1e-9, label="Nearest-Neighbor Chain Runtimes")
    plt.legend()
    plt.xlabel("Number of observations in dataset")
    plt.ylabel("Log(Runtime in Seconds)")
    plt.title("Ward Algorithm Runtime")
    plt.show()
