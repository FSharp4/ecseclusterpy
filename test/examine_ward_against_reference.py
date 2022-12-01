import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__ == "__main__":
    df = pd.read_csv("Timing_ward_comparative.csv")
    ward_profiling_data = df.to_numpy()
    n = int(np.sqrt(ward_profiling_data.shape[0]))
    reshaped_data = np.zeros([n, n, 7])
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
    ax.set_xlabel(df.columns[1])
    ax.set_ylabel(df.columns[2])
    ax.set_zlabel("Time to complete biclustering (sec)")
    fig.suptitle(df.columns[3])

    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax2.plot_surface(X, Y, reshaped_data[:, :, 4], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    ax2.set_xlabel(df.columns[1])
    ax2.set_ylabel(df.columns[2])
    ax2.set_zlabel("Time to complete biclustering (sec)")
    fig2.suptitle(df.columns[4])

    fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax3.plot_surface(X, Y, reshaped_data[:, :, 5], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig3.colorbar(surf, shrink=0.5, aspect=5)
    ax3.set_xlabel(df.columns[1])
    ax3.set_ylabel(df.columns[2])
    ax3.set_zlabel("Time to complete biclustering (sec)")
    fig3.suptitle(df.columns[6])

    fig4, ax4 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax4.plot_surface(X, Y, reshaped_data[:, :, 6], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig4.colorbar(surf, shrink=0.5, aspect=5)
    ax4.set_xlabel(df.columns[1])
    ax4.set_ylabel(df.columns[2])
    ax4.set_zlabel("Time to complete biclustering (sec)")
    fig4.suptitle(df.columns[5])
    plt.show()

    print("Debug Point")