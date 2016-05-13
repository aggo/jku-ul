#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 5

def read_data(filename):
    import csv, numpy as np
    instances = []
    localization_sites = []
    with open(filename, 'r') as csvfile:
        dataset = csv.reader(csvfile, delimiter=',')
        for row in dataset:
            localization_sites.append(row[-1])
            row = row[1:-1]  # remove id and localization site of the protein
            row = [float(i) for i in row]
            instances.append(row)
    size = len(instances)
    return instances, size, localization_sites

def normalize_data(data):
    from sklearn.preprocessing import normalize
    data = normalize(data, axis=0)
    return data

def center(data):
    from sklearn.preprocessing import normalize
    data = normalize(data, axis=0)
    return data

if __name__ == "__main__":
    import numpy as np, matplotlib.pyplot as plt

    # read the data from 1st file
    data, size, localization_sites = read_data('ecoli.csv')
    # get all the different localization sites
    loc_sites_set = set(localization_sites)
    print(localization_sites)
    print(loc_sites_set)
    # data = normalize_data(data)
    data -= np.mean(data, axis=0)

    # downproject
    from sklearn.decomposition import PCA
    pca_for_ecoli = PCA(n_components=2)
    pca_for_ecoli.fit(data)

    # compute the PC values
    data = pca_for_ecoli.transform(data)

    # plot data
    for loc in loc_sites_set:
        entries_for_current_loc = np.array([data[i] for i in range(len(data)-1) if localization_sites[i]==loc])
        plt.plot(entries_for_current_loc[:,0], entries_for_current_loc[:,1], 'o', label = loc)

    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Downprojected ecoli data")

    plt.show()
