import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("cutting_accuracies.csv")
    bicess_means = df.BICESS_mean.to_numpy()
    aicess_means = df.AICESS_mean.to_numpy()
    elbow_means = df.Elbow_mean.to_numpy()
    bicess_stdev = df.BICESS_std.to_numpy()
    aicess_stdev = df.AICESS_std.to_numpy()
    elbow_stdev = df.Elbow_std.to_numpy()

    ind = np.arange(len(bicess_means))
    width = 0.25
    x = np.arange(0.5, 4.5, 0.5)

    plt.figure()
    plt.errorbar(x, bicess_means, marker='o', yerr=bicess_stdev, label="BIC/ESS")
    plt.errorbar(x - 0.025, aicess_means, marker='o', yerr=aicess_stdev, label="AIC/ESS")
    plt.plot(x - 0.05, elbow_means, marker='o', label="Elbow")
    plt.ylim((0, 10))
    plt.xlabel("Cluster Standard Deviation (at Generation)")
    plt.ylabel("Predicted # Clusters by Dendrogram Cut")
    plt.title("Plot of Dendrogram-Cut Predicted # of Clusters in 5-cluster datasets")
    plt.legend()
    plt.grid(visible=True, which='major', axis='both')
    plt.show()
