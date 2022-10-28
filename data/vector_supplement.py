import numpy as np


def v_distance(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(p1 - p2), axis=1))


def s_distance_squared(p1, p2):
    return sum(np.square(p1 - p2))
