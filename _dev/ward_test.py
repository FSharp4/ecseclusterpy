import numpy as np
from scipy.cluster import hierarchy

from clustering import uniclustering
from data.generatedata import prebaked_data, prebaked_data2, generate


class WardTest:
    def __init__(self, data: np.ndarray):
        self.data = data

    def do(self):
        z1 = uniclustering.linkage(self.data)
        z2 = uniclustering.linkage(self.data, naive=True)
        return PASS if np.allclose(z1, z2) else FAIL


PASS = "Pass"
FAIL = "Fail"

if __name__ == "__main__":
    print(f"Test 1: {WardTest(prebaked_data().features).do()}")
    print(f"Test x: {WardTest(prebaked_data2()).do()}")
    little = generate(n_samples=10, n_features=3, centers=5, cluster_std=0.8, shuffle=True).features
    print(f"Test x: {WardTest(little).do()}")
    medium = generate(n_samples=50, n_features=5, centers=10, cluster_std=0.8, shuffle=True).features
    print(f"Test x: {WardTest(medium).do()}")
    big = generate(n_samples=100, n_features=10, centers=15, cluster_std=0.8, shuffle=True).features
    print(f"Test x: {WardTest(big).do()}")
    huge = generate(n_samples=500, n_features=15, centers=20, cluster_std=0.8, shuffle=True).features
    print(f"Test x: {WardTest(huge).do()}")
