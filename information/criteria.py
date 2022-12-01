import numpy as np

from clustering import uniclustering


def aic(data: np.ndarray) -> float:
    k = extract_parameter_count(data)
    t1: float = 2 * k
    t2, t4 = _calculate_information_criterion_elements(data)

    return t1 + t2 + t4


def bic(data: np.ndarray) -> float:
    """
    Calculates the Bayesian Information Criterion assuming a Gaussian
    distribution of datapoint deviations from cluster center.

    :param data: Datapoints within cluster
    :return: Value of cluster BIC
    """
    k = extract_parameter_count(data)
    t1: float = k * np.log(data.shape[0])
    t2, t4 = _calculate_information_criterion_elements(data)

    return t1 + t2 + t4


def _calculate_information_criterion_elements(data):
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    det_cov: float = np.linalg.det(cov)
    inv_cov: np.ndarray
    if det_cov > 0 and not np.allclose(det_cov, 0):
        inv_cov = np.linalg.inv(cov)
    else:
        inv_cov = np.linalg.pinv(cov)
        eigenvalues = np.linalg.eigvals(cov)
        det_cov = np.real(np.product(eigenvalues[eigenvalues > 1e-12]))
    t2: float = data.shape[0] * np.log(det_cov)
    # t3: float = n * m * np.log(2 * np.pi)
    # This is an omitted term in literature as it is constant in all but n,
    # thus the total sum remains constant over all iterations...

    t4: float = 0
    for point in data:
        residual: np.ndarray = np.atleast_2d(point - mean)
        t4 += np.matmul(residual, np.matmul(inv_cov, residual.T))[0, 0]
    return t2, t4


def extract_parameter_count(data: np.ndarray):
    m = data.shape[1]
    k = (m * (m + 1)) // 2 + m
    return k


# def bic_components(data: np.ndarray) -> np.ndarray:
#     """
#     Calculates the Bayesian Information Criterion assuming a Gaussian
#     distribution of datapoint deviations from cluster center.
#
#     :param data: Datapoints within cluster
#     :return: Value of cluster BIC
#     """
#     n = data.shape[0]
#     if n == 1:
#         return 0.0
#     m = data.shape[1]
#     k = (m * (m + 1)) // 2 + m
#     mean = np.mean(data, axis=0)
#     cov = np.cov(data.T)
#     det_cov: float = np.linalg.det(cov)
#     inv_cov: np.ndarray
#     if det_cov > 0 and not np.allclose(det_cov, 0):
#         inv_cov = np.linalg.inv(cov)
#     else:
#         inv_cov = np.linalg.pinv(cov)
#         eigenvalues = np.linalg.eigvals(cov)
#         det_cov = np.real(np.product(eigenvalues[eigenvalues > 1e-12]))
#
#     t1: float = k * np.log(n)
#     t2: float = n * np.log(det_cov)
#     t3: float = n * m * np.log(2 * np.pi)  # This is an omitted term in literature as it is constant in all but n
#     t4: float = 0
#     for point in data:
#         residual: np.ndarray = np.atleast_2d(point - mean)
#         t4 += np.matmul(residual, np.matmul(inv_cov, residual.T))[0, 0]
#
#     return np.array([t1, t2, t3, t4])


def calculate_aic_vector(data: np.ndarray, Z: np.ndarray):
    cluster_datapoint_map, n, merged_cluster_aic = _information_criterion_vector_setup(Z, data)

    for i in range(n, 2 * n - 1):
        points: np.ndarray = cluster_datapoint_map[i]
        merged_cluster_aic[i - n] = aic(data[points])

    aic_vector = _create_information_vector(Z, merged_cluster_aic, n)
    return aic_vector


def calculate_bic_vector(data: np.ndarray, Z: np.ndarray):
    cluster_datapoint_map, n, merged_cluster_bic = _information_criterion_vector_setup(Z, data)

    for i in range(n, 2 * n - 1):
        points: np.ndarray = cluster_datapoint_map[i]
        merged_cluster_bic[i - n] = bic(data[points])

    bic_vector = _create_information_vector(Z, merged_cluster_bic, n)
    return bic_vector


def _create_information_vector(Z, merged_cluster_bic, n):
    cluster_membership = np.vstack([np.arange(0, n)] * 2).T
    working_labels = cluster_membership[:, 1]
    ptr = n
    info_vector = np.zeros([n - 1])
    for (p1, p2) in Z[:, :2]:
        uniclustering.increment_point_membership(cluster_membership, p1, p2, ptr)
        merged_clusters = np.unique(working_labels)
        merged_clusters = merged_clusters[merged_clusters >= n]
        info_vector[ptr - n] = np.sum(merged_cluster_bic[merged_clusters - n])
        ptr += 1

    return info_vector


def _information_criterion_vector_setup(Z, data):
    n: int = data.shape[0]
    cluster_datapoint_map = uniclustering.get_cluster_datapoint_mapping(Z)
    merged_cluster_bic = np.zeros([n - 1])
    return cluster_datapoint_map, n, merged_cluster_bic

# def calculate_bic_vector_components(data: np.ndarray, Z: np.ndarray):
#     n: int = data.shape[0]
#     m: int = data.shape[1]
#     cluster_datapoint_map = uniclustering.get_cluster_datapoint_mapping(Z)
#     merged_cluster_bic = np.zeros([n - 1, 4])
#     for i in range(n, 2 * n - 1):
#         points: np.ndarray = cluster_datapoint_map[i]
#         datapoints: np.ndarray = data[points]
#         merged_cluster_bic[i - n] = bic(datapoints)
#         # if np.isnan(merged_cluster_bic[i - n]):
#         #     print("Debug NAN")
#
#     cluster_membership = np.vstack([np.arange(0, n)] * 2).T
#     working_labels = cluster_membership[:, 1]
#     ptr = n
#     bic_vector = np.zeros([n - 1], 4)
#     for (p1, p2) in Z[:, :2]:
#         uniclustering.increment_point_membership(cluster_membership, p1, p2, ptr)
#         merged_clusters = np.unique(working_labels)
#         merged_clusters = merged_clusters[merged_clusters >= n]
#         bic_vector[ptr - n] = np.sum(merged_cluster_bic[merged_clusters - n], axis=0)
#         ptr += 1
#     return bic_vector