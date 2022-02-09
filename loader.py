import csv

from generatedata import feature_headings, Dataset


def write(filename, data: Dataset):
    with open(filename, "w", newline='') as file:
        writer = csv.writer(file)
        headings = feature_headings(data.features[0].__len__())
        headings.append("Cluster Index")
        rows = [headings]
        for i in range(0, data.features.__len__()):
            feature = data.features[i]
            cluster = data.clusters[i]
            row = feature.tolist()
            row.append(cluster)
            rows.append(row)

        writer.writerows(rows)
