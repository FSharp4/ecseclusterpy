import numpy as np
import pandas as pd

from clustering import uniclustering
from data import generatedata
from information import dendrogram_cut

n = 500
m = 20
c1 = 5
c2 = 50
stdev1 = 1
stdev2 = 2.5

t = 30
folds = 10


def low_count_low_variance():
    global n, m, c1, c2, stdev1, stdev2
    return generatedata.generate(n, m, c1, stdev1, True).features


def low_count_high_variance():
    global n, m, c1, c2, stdev1, stdev2
    return generatedata.generate(n, m, c1, stdev2, True).features


def high_count_low_variance():
    global n, m, c1, c2, stdev1, stdev2
    return generatedata.generate(n, m, c2, stdev1, True).features


def high_count_high_variance():
    return generatedata.generate(n, m, c2, stdev2, True).features


if __name__ == "__main__":
    cross_validation_mean_predictions = np.ndarray([t, 4, 3])
    cross_validation_mean_stdevs = np.ndarray([t, 4, 3])
    raw_predictions = np.ndarray([t, 4, 3])
    for ii in range(t):
        print(f"Trial {ii}")
        data1 = low_count_low_variance()
        data2 = low_count_high_variance()
        data3 = high_count_low_variance()
        data4 = high_count_high_variance()

        folds1 = np.array(np.array_split(data1, folds))
        folds2 = np.array(np.array_split(data2, folds))
        folds3 = np.array(np.array_split(data3, folds))
        folds4 = np.array(np.array_split(data4, folds))

        bicess_sig1 = np.ndarray([folds])
        bicess_sig2 = np.ndarray([folds])
        bicess_sig3 = np.ndarray([folds])
        bicess_sig4 = np.ndarray([folds])

        aicess_sig1 = np.ndarray([folds])
        aicess_sig2 = np.ndarray([folds])
        aicess_sig3 = np.ndarray([folds])
        aicess_sig4 = np.ndarray([folds])

        elbow_sig1 = np.ndarray([folds])
        elbow_sig2 = np.ndarray([folds])
        elbow_sig3 = np.ndarray([folds])
        elbow_sig4 = np.ndarray([folds])

        for k in range(folds):
            print(f"Fold {k}")
            partial1 = np.vstack(np.delete(folds1, k, axis=0))
            partial2 = np.vstack(np.delete(folds2, k, axis=0))
            partial3 = np.vstack(np.delete(folds3, k, axis=0))
            partial4 = np.vstack(np.delete(folds4, k, axis=0))

            Z1 = uniclustering.ward_linkage(partial1)
            Z2 = uniclustering.ward_linkage(partial2)
            Z3 = uniclustering.ward_linkage(partial3)
            Z4 = uniclustering.ward_linkage(partial4)

            bicess_sig1[k] = dendrogram_cut.bicess_cut(partial1, Z1)
            bicess_sig2[k] = dendrogram_cut.bicess_cut(partial2, Z2)
            bicess_sig3[k] = dendrogram_cut.bicess_cut(partial3, Z3)
            bicess_sig4[k] = dendrogram_cut.bicess_cut(partial4, Z4)

            aicess_sig1[k] = dendrogram_cut.aicess_cut(partial1, Z1)
            aicess_sig2[k] = dendrogram_cut.aicess_cut(partial2, Z2)
            aicess_sig3[k] = dendrogram_cut.aicess_cut(partial3, Z3)
            aicess_sig4[k] = dendrogram_cut.aicess_cut(partial4, Z4)

            elbow_sig1[k] = dendrogram_cut.elbow_cut(partial1, Z1)
            elbow_sig2[k] = dendrogram_cut.elbow_cut(partial2, Z2)
            elbow_sig3[k] = dendrogram_cut.elbow_cut(partial3, Z3)
            elbow_sig4[k] = dendrogram_cut.elbow_cut(partial4, Z4)

        bicess_mean1 = np.mean(bicess_sig1)
        bicess_mean2 = np.mean(bicess_sig2)
        bicess_mean3 = np.mean(bicess_sig3)
        bicess_mean4 = np.mean(bicess_sig4)

        aicess_mean1 = np.mean(aicess_sig1)
        aicess_mean2 = np.mean(aicess_sig2)
        aicess_mean3 = np.mean(aicess_sig3)
        aicess_mean4 = np.mean(aicess_sig4)

        elbow_mean1 = np.mean(elbow_sig1)
        elbow_mean2 = np.mean(elbow_sig2)
        elbow_mean3 = np.mean(elbow_sig3)
        elbow_mean4 = np.mean(elbow_sig4)

        bicess_std1 = np.std(bicess_sig1)
        bicess_std2 = np.std(bicess_sig2)
        bicess_std3 = np.std(bicess_sig3)
        bicess_std4 = np.std(bicess_sig4)

        aicess_std1 = np.std(aicess_sig1)
        aicess_std2 = np.std(aicess_sig2)
        aicess_std3 = np.std(aicess_sig3)
        aicess_std4 = np.std(aicess_sig4)

        elbow_std1 = np.std(elbow_sig1)
        elbow_std2 = np.std(elbow_sig2)
        elbow_std3 = np.std(elbow_sig3)
        elbow_std4 = np.std(elbow_sig4)

        cross_validation_mean_predictions[ii] = np.array([
            [bicess_mean1, aicess_mean1, elbow_mean1],
            [bicess_mean2, aicess_mean2, elbow_mean2],
            [bicess_mean3, aicess_mean3, elbow_mean3],
            [bicess_mean4, aicess_mean4, elbow_mean4],
        ])

        cross_validation_mean_stdevs[ii] = np.array([
            [bicess_std1, aicess_std1, elbow_std1],
            [bicess_std2, aicess_std2, elbow_std2],
            [bicess_std3, aicess_std3, elbow_std3],
            [bicess_std4, aicess_std4, elbow_std4],
        ])

        Z1 = uniclustering.ward_linkage(data1)
        Z2 = uniclustering.ward_linkage(data2)
        Z3 = uniclustering.ward_linkage(data3)
        Z4 = uniclustering.ward_linkage(data4)

        raw_predictions[ii, 0, 0] = dendrogram_cut.bicess_cut(data1, Z1)
        raw_predictions[ii, 1, 0] = dendrogram_cut.bicess_cut(data2, Z2)
        raw_predictions[ii, 2, 0] = dendrogram_cut.bicess_cut(data3, Z3)
        raw_predictions[ii, 3, 0] = dendrogram_cut.bicess_cut(data4, Z4)

        raw_predictions[ii, 0, 1] = dendrogram_cut.aicess_cut(data1, Z1)
        raw_predictions[ii, 1, 1] = dendrogram_cut.aicess_cut(data2, Z2)
        raw_predictions[ii, 2, 1] = dendrogram_cut.aicess_cut(data3, Z3)
        raw_predictions[ii, 3, 1] = dendrogram_cut.aicess_cut(data4, Z4)

        raw_predictions[ii, 0, 2] = dendrogram_cut.elbow_cut(data1, Z1)
        raw_predictions[ii, 1, 2] = dendrogram_cut.elbow_cut(data2, Z2)
        raw_predictions[ii, 2, 2] = dendrogram_cut.elbow_cut(data3, Z3)
        raw_predictions[ii, 3, 2] = dendrogram_cut.elbow_cut(data4, Z4)

        # raw_predictions[ii] = [
        #     [raw_bicess1, raw_aicess1, raw_elbow1],
        #     [raw_bicess2, raw_aicess2, raw_elbow2],
        #     [raw_bicess3, raw_aicess3, raw_elbow3],
        #     [raw_bicess4, raw_aicess4, raw_elbow4],
        # ]

    mean_raw_predictions = np.mean(raw_predictions, axis=0)
    raw_prediction_stdev = np.std(raw_predictions, axis=0)

    means = pd.DataFrame(data=mean_raw_predictions,
                         columns=["BIC/ESS", "AIC/ESS", "Elbow"],
                         index=["Few Clusters/Low Variance", "Few Clusters/High Variance", "Many Clusters/Low Variance",
                                "Many Clusters/High Variance"])

    stdevs = pd.DataFrame(raw_prediction_stdev,
                          columns=["BIC/ESS", "AIC/ESS", "Elbow"],
                          index=["Few Clusters/Low Variance", "Few Clusters/High Variance",
                                 "Many Clusters/Low Variance",
                                 "Many Clusters/High Variance"])

    print("Mean Predicted # of Clusters")
    print(means)

    print("# Of Clusters Prediction Standard Deviation")
    print(stdevs)
