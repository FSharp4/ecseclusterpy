import numpy as np


def variance(dataset: np.ndarray) -> float:
    e_x = np.mean(dataset, axis=0)
    e_x2 = np.mean(np.square(dataset), axis=0)
    var = e_x2 - np.square(e_x)



