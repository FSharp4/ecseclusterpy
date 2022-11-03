import numpy as np
from scipy.spatial.distance import pdist, squareform

if __name__ == "__main__":
    test = np.array([[1, 3, 2, 4], [2, 3, 5, 8], [3, 1, 4, 1], [5, 1, 1, 1]])
    dm = squareform(pdist(test))
    dmt = squareform(pdist(test.transpose()))
    print("Debug Point")