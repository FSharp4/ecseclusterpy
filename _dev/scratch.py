from sklearn.datasets import load_iris

from data.generatedata import RAW_generate_example

iris = load_iris().data
example = RAW_generate_example()

print("Debug point")
