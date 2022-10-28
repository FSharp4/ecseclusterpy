import csv

import matplotlib.pyplot as plt
import numpy

from data import generatedata


def read(filename) -> generatedata.Dataset:
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        heading_row = next(reader)
        headings = heading_row[0:heading_row.__len__() - 1]
        row = next(reader)
        end = row.__len__() - 1
        features = [numpy.float64(row[0:end])]
        clusters = [numpy.int64(row[end])]
        for row in reader:
            features.append(numpy.float64(row[0:end]))
            clusters.append(numpy.int64(row[end]))

        features = numpy.array(features)
        clusters = numpy.array(clusters)

        reconstructed_data = generatedata.Dataset(features, clusters)
        reconstructed_data.headings = headings
        return reconstructed_data


if __name__ == '__main__':
    data = read("example.csv")
    plt.scatter(data.features[:, 0], data.features[:, 1])
    plt.show()
