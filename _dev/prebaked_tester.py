from scipy.spatial.distance import squareform

import generatedata
import nnchain

def do():
    data = generatedata.prebaked_data().features
    distance_matrix = squareform(nnchain._pdist(data))
    print("Debug Point")

do()