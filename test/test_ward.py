import numpy as np
import unittest
from scipy.cluster import hierarchy

from clustering import uniclustering
from data.generatedata import generate_gan, prebaked_data2, generate


class WardTest():
    def __init__(self, data: np.ndarray):
        self.data = data

    def opt_eq_naive(self):
        z1 = uniclustering.ward_linkage(self.data)
        z2 = uniclustering.ward_linkage(self.data, naive=True)
        return np.allclose(z1, z2)

    def opt_eq_ref(self):
        z1 = uniclustering.ward_linkage(self.data)
        z2 = hierarchy.linkage(self.data, method="ward")
        return np.allclose(z1, z2)


little = generate(n_samples=10, n_features=3, centers=5, cluster_std=0.8, shuffle=True).features
medium = generate(n_samples=50, n_features=5, centers=10, cluster_std=0.8, shuffle=True).features
big = generate(n_samples=100, n_features=10, centers=15, cluster_std=0.8, shuffle=True).features
huge = generate(n_samples=500, n_features=15, centers=20, cluster_std=0.8, shuffle=True).features
pb1 = generate_gan().features
pb2 = prebaked_data2()


class TestWardMethods(unittest.TestCase):

    def setUp(self) -> None:
        self.little_tester = WardTest(little)
        self.medium_tester = WardTest(medium)
        self.big_tester = WardTest(big)
        self.huge_tester = WardTest(huge)
        self.pb1_tester = WardTest(pb1)
        self.pb2_tester = WardTest(pb2)

    def test_little_opt_naive(self):
        self.assertTrue(self.little_tester.opt_eq_naive())

    def test_little_opt_ref(self):
        self.assertTrue(self.little_tester.opt_eq_ref())

    def test_medium_opt_naive(self):
        self.assertTrue(self.medium_tester.opt_eq_naive())

    def test_medium_opt_ref(self):
        self.assertTrue(self.medium_tester.opt_eq_ref())

    def test_big_opt_naive(self):
        self.assertTrue(self.big_tester.opt_eq_naive())

    def test_big_opt_ref(self):
        self.assertTrue(self.big_tester.opt_eq_ref())

    def test_huge_opt_naive(self):
        self.assertTrue(self.huge_tester.opt_eq_naive())

    def test_huge_opt_ref(self):
        self.assertTrue(self.huge_tester.opt_eq_ref())

    def test_pb1_opt_naive(self):
        self.assertTrue(self.pb1_tester.opt_eq_naive())

    def test_pb1_opt_ref(self):
        self.assertTrue(self.pb1_tester.opt_eq_ref())

    def test_pb2_opt_naive(self):
        self.assertTrue(self.pb2_tester.opt_eq_naive())

    def test_pb2_opt_ref(self):
        self.assertTrue(self.pb2_tester.opt_eq_ref())


# if __name__ == "__main__":
#     unittest.main()
