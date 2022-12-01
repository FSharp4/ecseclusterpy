from clustering import biclustering
from data import generatedata

if __name__ == "__main__":
    data = generatedata.prebaked_data2()
    Z1_o, Z1_f = biclustering.custom_optimized(data)
    Z2_o, Z2_f, om_2 = biclustering.custom_naive(data)
    print("Debug point")
