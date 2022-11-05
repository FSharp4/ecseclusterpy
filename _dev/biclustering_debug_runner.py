from clustering import biclustering
from data import generatedata

if __name__ == "__main__":
    data = generatedata.generate_gan().features
    Z_o, Z_f, o_m = biclustering.naive_integrated_slow(data)
    Z_o2, Z_f2 = biclustering.naive_separated(data)
    print("Debug point")
