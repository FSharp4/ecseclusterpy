from scipy.cluster.hierarchy import linkage as reference_ward
from clustering.uniclustering import linkage as nnchain_ward
from data.generatedata import prebaked_data2_set as dataset
from data import matrix_logger

if __name__ == "__main__":
    data = dataset().features
    matrix_logger.change_series("reference_distance_matrix")
    Z1 = reference_ward(data, method="ward")
    matrix_logger.change_series("nnchain_distance_matrix")
    Z2 = nnchain_ward(data)
    store = matrix_logger.store
    print("Debug Point")
