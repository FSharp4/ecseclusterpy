import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__ == "__main__":
    ward_profiling_data = pd.read_csv("Timing_ward_optimized.csv").to_numpy()
    n = int(np.sqrt(ward_profiling_data.shape[0]))
    reshaped_data = np.zeros([n, n, 5])
    values = np.unique(ward_profiling_data[:, 1])
    for i in range(len(values)):
        value = values[i]
        reshaped_data[i, :, :] = ward_profiling_data[ward_profiling_data[:, 1] == value]

    X = reshaped_data[0, :, 2]
    Y = X
    X, Y = np.meshgrid(X, Y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, reshaped_data[:, :, 3], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax2.plot_surface(X, Y, reshaped_data[:, :, 4], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    print("Debug Point")