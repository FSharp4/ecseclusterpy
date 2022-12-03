import numpy as np
from scipy.stats import norm

from clustering import variance
from information import criteria


def scheme1(data: np.ndarray, Z: np.ndarray):
    x = np.gradient(Z[:, 2])
    y = np.gradient(criteria.calculate_bic_vector(data, Z))
    return _prototype_measure_2(x, y)


def scheme2(data: np.ndarray, Z: np.ndarray):
    x = np.gradient(Z[:, 2])
    y = np.gradient(criteria.calculate_aic_vector(data, Z))
    return _prototype_measure_2(x, y)


def scheme3(data: np.ndarray, Z: np.ndarray):
    x = Z[:, 2]
    y = np.gradient(criteria.calculate_bic_vector(data, Z))

    return _prototype_measure_2(x, y, threshold=0.95)


def bicess_cut(data: np.ndarray, Z: np.ndarray):
    x = Z[:, 2]
    y = np.gradient(criteria.calculate_bic_vector(data, Z))

    return _prototype_measure(x, y, threshold=0.9)


def aicess_cut(data: np.ndarray, Z: np.ndarray):
    x = Z[:, 2]
    y = np.gradient(criteria.calculate_aic_vector(data, Z))
    return _prototype_measure(x, y, threshold=0.9)


def _prototype_measure(x_measure, y_measure, threshold=0.95):
    r = np.linalg.norm(np.vstack([x_measure, y_measure]), axis=0)
    mu_r = np.mean(r)
    std_r = np.std(r)
    z = (r - mu_r) / std_r
    p = norm.cdf(z)
    significant_clusters = 0
    for i in range(len(p)):
        if p[-(i + 1)] < threshold:
            significant_clusters = i + 1
            break

    return significant_clusters


def _prototype_measure_2(x_measure, y_measure, threshold: float = 0.95):
    r = np.linalg.norm(np.vstack([x_measure, y_measure]), axis=0)
    mu_r = np.mean(r)
    std_r = np.std(r)
    z = (r - mu_r) / std_r
    p = norm.cdf(z)
    significant_cluster = p.shape[0] - np.argwhere(p >= threshold)[0] + 1

    return significant_cluster


def curvature(x: np.ndarray, y: np.ndarray):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    v = np.vstack([dx_dt, dy_dt])
    ds_dt = np.linalg.norm(v, axis=0)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (np.power(ds_dt, 3))
    return curvature


def elbow_cut(data: np.ndarray, Z: np.ndarray):
    x = np.arange(1, data.shape[0])
    ess = variance.ward_ess(Z[:, 2])
    y = -(ess[::-1] - ess[-1])
    k = curvature(x, y)
    index = np.argmax(k)
    return x[index]


def find_cutting_distance(Z: np.ndarray, significant_clusters: int):
    low_d = Z[:, -2][-significant_clusters]
    high_d = Z[:, -2][-significant_clusters + 1]
    return (high_d + low_d) / 2
