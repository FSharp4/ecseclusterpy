import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

from clustering import uniclustering
from data.generatedata import generate_gan

if __name__ == "__main__":
    smallest = generate_gan().features
    Z = hierarchy.linkage(smallest, method="ward")
    plt.figure()
    hierarchy.dendrogram(Z)
    plt.xlabel("Datapoint Index")
    plt.ylabel("(Dis)similarity Metric")
    plt.title("Example Dendrogram")
    plt.show()

