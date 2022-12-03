import random

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy

from clustering import biclustering
from data import generatedata

if __name__ == "__main__":
    n = 200
    c = 10
    t = 30

    # matches_naive_obs = np.ndarray([t], dtype=bool)
    # matches_naive_fea = np.ndarray([t], dtype=bool)
    #
    # for ii in range(t):
    #     print(f"{ii}")
    #     data = generatedata.generate(random.randint(50, n),
    #                                  random.randint(50, n),
    #                                  random.randint(2, c),
    #                                  random.uniform(0.5, 2.0),
    #                                  True).features
    #     Z_o, Z_f, _ = biclustering.custom_naive(data)
    #     Z_o2, Z_f2 = biclustering.custom_optimized(data)
    #     matches_naive_obs = np.allclose(Z_o, Z_o2)
    #     matches_naive_fea = np.allclose(Z_f, Z_f2)
    #
    # print(matches_naive_fea)
    # print(matches_naive_obs)

    data = generatedata.generate(25, 25, 4, 1.5, True).features
    Z_o, Z_f = biclustering.custom_optimized(data)
    Z_o2, Z_f2, _ = biclustering.custom_naive(data)

    fig = plt.figure()
    fg = seaborn.clustermap(data, row_linkage=Z_o, col_linkage=Z_f)
    plt.title("   (Decoupled)")
    plt.show()

    plt.figure()
    seaborn.clustermap(data, row_linkage=Z_o2, col_linkage=Z_f2)
    plt.title("(Coupled)")
    plt.show()

    # SMALL_SIZE = 16
    # MEDIUM_SIZE = 24
    # BIGGER_SIZE = 32
    #
    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    fig.suptitle("Dendrograms for Observations/Features for Coupled and Decoupled Ward Biclustering")
    plt.subplot(2, 2, 1)
    hierarchy.dendrogram(Z_o, no_labels=True)
    plt.title("Observations, Decoupled")
    plt.ylabel("Net ESS Increase")
    plt.subplot(2, 2, 2)
    hierarchy.dendrogram(Z_o2, no_labels=True)
    plt.title("Observations, Coupled")
    plt.ylabel("Net ESS Increase")
    plt.subplot(2, 2, 3)
    hierarchy.dendrogram(Z_f, no_labels=True)
    plt.title("Features, Decoupled")
    plt.ylabel("Net ESS Increase")
    plt.subplot(2, 2, 4)
    hierarchy.dendrogram(Z_f2, no_labels=True)
    plt.title("Features, Coupled")
    plt.ylabel("Net ESS Increase")
    plt.show()

