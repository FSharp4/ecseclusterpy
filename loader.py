import csv

from generatedata import feature_headings, Dataset


def write(filename, data: Dataset):
    with open(filename, "w", newline='') as file:
        writer = csv.writer(file)
        headings = feature_headings(data.features[0].__len__())
        rows = [ headings ]
        for feature in data.features:
            rows.append(feature)

        writer.writerows(rows)

