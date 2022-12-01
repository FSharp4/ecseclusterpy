import time
from pathlib import Path

import numpy as np
import pandas as pd

from clustering import uniclustering, biclustering
from data.generatedata import generate_simple

if __name__ == '__main__':
    n_points = np.linspace(10, 1000, 30, endpoint=True, dtype=int)
    n_features = n_points
    Path("profiling/ward").mkdir(parents=True, exist_ok=True)
    timing = np.ndarray([n_points.size * n_features.size, 4])
    ptr = 0
    for i in range(n_points.size):
        for j in range(n_features.size):
            print(f"({n_points[i]}, {n_features[j]})")
            data = generate_simple(n_samples=n_points[i], n_features=n_features[j]).features
            t0 = time.time()
            uniclustering.ward_linkage(data, naive=False)
            t1 = time.time()
            biclustering.custom_optimized(data)
            t2 = time.time()
            timing[ptr] = [n_points[i], n_features[j], t1-t0, t2-t1]
            ptr += 1

    dataframe = pd.DataFrame(data=timing,
                             columns=["Points", "Features", "Optimized Uniclustering",
                                      "Optimized Biclustering"])
    dataframe.to_csv("Timing_ward_optimized.csv")
