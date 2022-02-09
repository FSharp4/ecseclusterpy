import csv


def write(filename, data_export):
    with open(filename, "w", newline='') as file:
        writer = csv.writer(file)
        headings = data_export.headings
        headings.append("Cluster Index")
        rows = [headings]
        for i in range(0, data_export.features.__len__()):
            feature = data_export.features[i]
            cluster = data_export.clusters[i]
            row = feature.tolist()
            row.append(cluster)
            rows.append(row)

        writer.writerows(rows)
