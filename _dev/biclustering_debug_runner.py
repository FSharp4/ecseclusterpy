from clustering import biclustering
from data import generatedata

if __name__ == "__main__":
    data = generatedata.generate_gan().features
    biclustering.naive_integrated_slow(data)
