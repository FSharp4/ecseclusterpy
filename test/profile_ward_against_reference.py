import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

from clustering import uniclustering, biclustering
from data.generatedata import generate_simple

if __name__ == '__main__':
    n_points = np.linspace(10, 1000, 30, endpoint=True, dtype=int)
    n_features = n_points
    Path("profiling/ward").mkdir(parents=True, exist_ok=True)
    timing = np.ndarray([n_points.size * n_features.size, 6])
    ptr = 0
    for i in range(n_points.size):
        for j in range(n_features.size):
            data = generate_simple(n_samples=n_points[i], n_features=n_features[j]).features
            t0 = time.time()
            hierarchy.linkage(data, method="ward")
            t1 = time.time()
            uniclustering.ward_linkage(data, naive=False)
            t2 = time.time()
            biclustering.custom_optimized(data)
            t3 = time.time()
            biclustering.reference_optimized(data)
            t4 = time.time()
            timing[ptr] = [n_points[i], n_features[j], t1 - t0, t2 - t1, t3 - t2, t4 - t3]
            ptr += 1

    dataframe = pd.DataFrame(data=timing,
                             columns=["Points", "Features", "Reference Uniclustering", "Optimized Uniclustering",
                                                                                       "Reference Biclustering",
                                      "Optimized Biclustering"])
    dataframe.to_csv("Timing_ward_comparative.csv")
