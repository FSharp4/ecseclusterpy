from ward import _debug_ward as reference_ward
from nnchain import ward as nnchain_ward
from generatedata import prebaked_data2_set as dataset
import matrix_logger

if __name__ == "__main__":
    data = dataset()
    matrix_logger.change_series("reference_distance_matrix")
    Z1 = reference_ward(data)
    matrix_logger.change_series("nnchain_distance_matrix")
    Z2 = nnchain_ward(data.features)
    store = matrix_logger.store
    print("Debug Point")
