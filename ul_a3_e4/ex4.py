#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 4

import math

import sklearn


def read_data(filename):
    import csv, numpy as np
    instances = []
    with open(filename, 'r') as csvfile:
        dataset = csv.reader(csvfile, delimiter=',')
        for row in dataset:
            row = [float(i) for i in row]
            instances.append(row)
    size = len(instances)
    return instances, size


def plot_in_3d(x, y, z, title):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', s=20, c='b', depthshade=True)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_zlabel("Z coordinate")
    plt.title(title)
    plt.show()

def plot_in_2d(x, y, title):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    plt.plot(x, y, 'b.')
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title(title)
    plt.show()


def get_normalized_vectors_by_coordinates_3d(data):
    # returns 3 vectors with the 3 coordinates
    y = [entry[1] for entry in data]
    z = [entry[2] for entry in data]
    x = [entry[0] for entry in data]
    avgx = np.mean(x)
    avgy = np.mean(y)
    avgz = np.mean(z)
    x = [entry - avgx for entry in x]
    y = [entry - avgy for entry in y]
    z = [entry - avgz for entry in z]
    return x, y, z


def get_normalized_vectors_by_coordinates_2d(data):
    # returns 2 vectors with the 2 coordinates
    y = [entry[1] for entry in data]
    x = [entry[0] for entry in data]
    avgx = np.mean(x)
    avgy = np.mean(y)
    x = [entry - avgx for entry in x]
    y = [entry - avgy for entry in y]
    return x, y

if __name__ == "__main__":
    import numpy as np, matplotlib.pyplot as plt

    # read the data from 1st file
    data, size = read_data('pca1.csv')
    x, y, z = get_normalized_vectors_by_coordinates_3d(data)
    plot_in_3d(x, y, z, "pca1.csv data")

    # read the data from 2nd file
    data2, size = read_data('pca2.csv')
    x2, y2 = get_normalized_vectors_by_coordinates_2d(data2)
    plot_in_2d(x2, y2, "pca2.csv data")

    # apply pca to get the first 2 components for first dataset
    from sklearn.decomposition import PCA
    pca_for_ds1 = PCA(n_components=2)
    pca_for_ds1.fit(data)

    print("First component for dataset1:")
    print(pca_for_ds1.components_[0])
    print("Percent of data explained by the first component for ds1:")
    print(pca_for_ds1.explained_variance_ratio_[0])

    # apply pca to get the first 1 components for first dataset
    pca_for_ds2 = PCA(n_components=1)
    pca_for_ds2.fit(data2)

    print("First component for dataset2:")
    print(pca_for_ds2.components_[0])
    print("Percent of data explained by the first component for ds2:")
    print(pca_for_ds2.explained_variance_ratio_[0])
