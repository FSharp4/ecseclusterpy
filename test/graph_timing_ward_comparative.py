import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    points = np.array([180, 180, 1000, 1000])
    features = np.array([180, 1000, 180, 1000])
    reference_uniclustering = np.array([0.001617908, 0.006417036, 0.052174091, 0.251837969])
    custom_uniclustering = np.array([0.037990808, 0.044503927, 0.895148993, 1.035551071])
    data = np.vstack([reference_uniclustering, custom_uniclustering]).T.tolist()
    columns = ("(P, F) = (180, 180)", "(P, F) = (180, 1000)", "(P, F) = (1000, 180)", "(P, F) = (1000, 1000)")
    rows = ("Reference Implementation", "Custom Implementation")
    bar_width = 0.33
    br1 = np.arange(len(reference_uniclustering))
    br2 = [x + bar_width for x in br1]
    # br3 = [x + bar_width for x in br2]

    plt.figure()
    plt.bar(br1, np.log(reference_uniclustering), color='r', width=bar_width, edgecolor='grey', label='Reference Implementation')
    plt.bar(br2, np.log(custom_uniclustering), color='b', width=bar_width, edgecolor='grey', label='Custom Implementation')
    plt.ylabel("Runtime in Seconds")
    plt.title("Runtime behavior of custom and Scipy Ward clustering implementations")
    plt.legend()
    plt.show()

    # colors = plt.cm.BuPu(np.linspace(0.1, 0.5, len(rows)))
    # n_rows = len(data)
    # index = np.arange(len(columns)) + 0.3
    # bar_width = 0.4
    #
    # y_offset = np.zeros(len(columns))
    #
    # cell_text = []
    # for row in range(n_rows):
    #     plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    #     y_offset = y_offset + data[row]
    #     cell_text.append(['%1.4f' % x for x in y_offset])
    #
    # colors = colors[::-1]
    # cell_text.reverse()
    #
    # plt.ylabel("Runtime (Seconds)")
    # plt.title("Runtime behavior of custom and Scipy Ward clustering implementations")
    # plt.show()