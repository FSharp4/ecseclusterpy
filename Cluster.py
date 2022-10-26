import warnings
from math import sqrt, fabs
from numbers import Number
from typing import Union

import numpy as np


# @DeprecationWarning
class Cluster:
    index: Number = -1  # Sentinel for 'not set'
    points: any
    mean: any
    size: Number  # FIXME not safe
    singleDim: bool

    def __init__(self, points: any, is_single_dim=False):
        self.singleDim = is_single_dim
        self.points = points
        tmp = self.points[0]
        ndim = 1
        if isinstance(tmp, np.ndarray):
            ndim = tmp.__len__()
        elif not is_single_dim:
            ndim = points.__len__()
            self.points = np.ndarray([1, ndim])
            self.points[0] = points
        else:
            mean = sum(points)
            return

        self.mean = np.ndarray([ndim])
        for dim in range(ndim):
            dimensional_value_sum = 0
            for point in self.points:
                dimensional_value_sum = dimensional_value_sum + point[dim]

            self.mean[dim] = dimensional_value_sum / self.points.__len__()

        self.size = self.points.__len__()

    def distance_to(self, position: Union[np.ndarray, Number]) -> float:
        distance: float = 0
        if isinstance(position, np.ndarray) and isinstance(self.mean, np.ndarray):
            for i in range(position.__len__()):
                distance = distance + (position[i] - self.mean[i]) ** 2

            distance = sqrt(distance)
        elif isinstance(position, Number) and isinstance(self.mean, Number):  # single dimension case
            distance = fabs(position - self.mean)

        return distance


def merge(c1, c2) -> Cluster:
    points = np.ndarray([c1.points.__len__() + c2.points.__len__(), c2.points[0].__len__()])
    if c1.points.__len__() != 0:
        points[0:c1.points.__len__()] = c1.points

    if c2.points.__len__() != 0:
        points[c1.points.__len__():c1.points.__len__() + c2.points.__len__()] = c2.points

    return Cluster(points)


def v_distance(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(p1 - p2), axis=1))


def s_distance_squared(p1, p2):
    return sum(np.square(p1 - p2))
