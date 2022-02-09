# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import seaborn

# import generatedata
# import reader

# read_from_existing = True
import generatedata


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # data = None
    # if read_from_existing:
    #     data = reader.read("example.csv")
    # else:
    #     data = generatedata.generate_example()

    data = generatedata.generate(n_samples=200, n_features=2, centers=5, cluster_std=0.6, shuffle=True)
    test_plot = seaborn.clustermap(data.features, method="ward")
    # test_plot.savefig("test_plot.png")
    print("Done")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
