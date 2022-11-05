import time
from pathlib import Path

import numpy as np
import pandas as pd

from clustering import uniclustering, biclustering
from data.generatedata import generate_simple

if __name__ == '__main__':
    n_points = np.logspace(1, 3, num=5, endpoint=True, dtype=int)
    n_features = np.logspace(1, 3, num=5, endpoint=True, dtype=int)
    Path("profiling/ward").mkdir(parents=True, exist_ok=True)
    timing = np.ndarray([n_points.size * n_features.size, 6])
    ptr = 0
    for i in range(n_points.size):
        for j in range(n_features.size):
            print(f"({n_points[i]}, {n_features[j]})")
            data = generate_simple(n_samples=n_points[i], n_features=n_features[j]).features
            t0 = time.time()
            uniclustering.linkage(data, naive=True)
            t1 = time.time()
            uniclustering.linkage(data, naive=False)
            t2 = time.time()
            biclustering.naive_integrated_slow(data)
            t3 = time.time()
            biclustering.naive_separated(data)
            t4 = time.time()
            timing[ptr] = [n_points[i], n_features[i], t1 - t0, t2 - t1, t3 - t2, t4 - t3]
            ptr += 1

    dataframe = pd.DataFrame(data=timing,
                             columns=["Points", "Features", "Optimized Uniclustering", "Naive Uniclustering",
                                      "Optimized Biclustering", "Naive Biclustering"])
    dataframe.to_csv("Timing.csv")
