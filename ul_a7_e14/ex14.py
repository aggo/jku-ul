#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 14

import itertools
import numpy as np
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Helpers --------------------------------------------------------------------------------------------------------------
def read_file(filename):
    csvdata = np.genfromtxt(filename, delimiter=',')
    if len(csvdata.shape) > 1:
        csvdata = csvdata[1:, :]  # remove the first line, the header
    else:
        csvdata = csvdata[1:]
    return csvdata


def computeDistance(chosen_centroid, point):
    xc = chosen_centroid[0]
    yc = chosen_centroid[1]

    xp = point[0]
    yp = point[1]

    dist = np.sqrt((xc - xp) ** 2 + (yc - yp) ** 2)
    return dist


def build_centroids_map(nr_clusters):
    points_location = np.random.rand(nr_clusters, 2)  # we have 2D data
    centroids_map = {}
    for i in range(nr_clusters):
        centroids_map[i] = points_location[i]
    return centroids_map


def build_update_map_for_centroids(nr_clusters):
    update_map = {}
    for i in range(nr_clusters):
        update_map[i] = [0, 0, 0]  # 3 values: 1st for the sum of X1s, 2nd for the sum of X2s, 3rd for # of terms in sum
        # the new coordinates will be be computed by dividing the sums to the number
    return update_map


def build_centroids_to_assigned_points_map(nr_clusters):
    centroids_to_assigned_points_map = {}
    for i in range(nr_clusters):
        centroids_to_assigned_points_map[
            i] = []  # 3 values: 1st for the sum of X1s, 2nd for the sum of X2s, 3rd for # of terms in sum
        # the new coordinates will be be computed by dividing the sums to the number
    return centroids_to_assigned_points_map


# Plotting -------------------------------------------------------------------------------------------------------------
def plot_in_2d_initial_data(points, labels, title):
    cluster_colors = itertools.cycle([colors[labels[i] * 50] for i in range(len(labels))])

    plot_matrix_size = np.math.ceil(np.math.sqrt(NR_CELLS_COMBINED_PLOT))
    plt.subplot(plot_matrix_size, plot_matrix_size, INDEX_IN_COMBINED_PLOT)

    for point in points:
        plt.scatter(point[0], point[1], color=next(cluster_colors))
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])
    plt.title("Correct splitting")

    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.draw()


def plot_in_2d_each_cluster(centroids_to_assigned_points_map, centroids):
    ax1.clear()
    for centroid in centroids_to_assigned_points_map.iterkeys():
        points = centroids_to_assigned_points_map[centroid]
        ax1.scatter([i[0] for i in points], [i[1] for i in points], marker='o', color=colors[centroid * 50], )
        ax1.scatter(centroids[centroid][0], centroids[centroid][1], marker='o', s=81, linewidths=2,
                    color='white', zorder=10, edgecolors="black")
    plt.draw()
    plt.pause(0.05)


def plot_in_combined_plot(centroids_to_assigned_points_map, centroids):
    plot_matrix_size = np.math.ceil(np.math.sqrt(NR_CELLS_COMBINED_PLOT))
    plt.subplot(plot_matrix_size, plot_matrix_size, INDEX_IN_COMBINED_PLOT)
    for centroid in centroids_to_assigned_points_map.iterkeys():
        points = centroids_to_assigned_points_map[centroid]
        plt.plot([i[0] for i in points], [i[1] for i in points], '.')
        plt.scatter(centroids[centroid][0], centroids[centroid][1], marker='o', s=169, linewidths=3,
                    color='white', zorder=10, edgecolors="black")
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title("C:" + str(nr_clusters) + " E" + str(experiment))
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])


# KMEANS -_-------------------------------------------------------------------------------------------------------------
def kmeans(nr_clusters, plot_after_each_run=True, plot_all_in_one_graph=True):

    global INDEX_IN_COMBINED_PLOT
    centroids = build_centroids_map(nr_clusters)
    nr_runs = 1

    while (nr_runs <= MAX_ITERATIONS):
        update_map_for_centroids = build_update_map_for_centroids(nr_clusters)
        centroids_to_assigned_points_map = build_centroids_to_assigned_points_map(nr_clusters)
        for point in x_normalized:
            min_dist = 1000000;
            chosen_centroid = None
            for centroid in centroids.iteritems():
                current_dist = computeDistance(centroid[1], point)
                if current_dist < min_dist:
                    chosen_centroid = centroid[0]
                    min_dist = current_dist

            # to compute the new locations of the centroids,
            # when a point is assigned to them, compute a sum
            # and increment a counter. then, when the time comes
            # to compute the new location, simply divide the 2 values
            update_map_for_centroids[chosen_centroid][0] += point[0]
            update_map_for_centroids[chosen_centroid][1] += point[1]
            update_map_for_centroids[chosen_centroid][2] += 1
            centroids_to_assigned_points_map[chosen_centroid].append(point)

        # compute new centroids centers
        for centroid in centroids.iteritems():
            update_map_current_centroid = update_map_for_centroids[centroid[0]]
            if update_map_current_centroid[2] != 0:  # if current centroid has no points assigned
                centroid[1][0] = update_map_current_centroid[0] / update_map_current_centroid[2]
                centroid[1][1] = update_map_current_centroid[1] / update_map_current_centroid[2]
            else:
                centroid[1][0] = 0
                centroid[1][1] = 0

            update_map_for_centroids[centroid[0]] = [0, 0, 0]  # reinitialize the map for the next round of point distance comp
        if plot_after_each_run:
            plot_in_2d_each_cluster(centroids_to_assigned_points_map, centroids)
        nr_runs += 1

    if plot_all_in_one_graph:
        plot_in_combined_plot(centroids_to_assigned_points_map, centroids)
        INDEX_IN_COMBINED_PLOT += 1
    plot_in_2d_each_cluster(centroids_to_assigned_points_map, centroids)


# ---------------------------------------------------------------------------------------------------------------
# Some constants
MAX_ITERATIONS = 20
NR_EXPERIMENTS_WITH_EACH_CLUSTER = 4
NR_CLUSTERS_LIST = [2, 3, 4, 5, 6]

# Reading the data
data = read_file("kmeansdata.csv")
x = data[:, :-1]
x_normalized = scale(x)
y = data[:, -1]

# Some plotting setup
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
colors = cm.rainbow(np.linspace(0, 1, len(y)))

# Run and notice how the cluster centers are computed
kmeans(nr_clusters=4, plot_after_each_run=True, plot_all_in_one_graph=False)

# Plot correct splitting followed by computed splitting

# We'll have a grid with all the experiments for all the numbers of clusters + the initial data
INDEX_IN_COMBINED_PLOT = 1
NR_CELLS_COMBINED_PLOT = len(NR_CLUSTERS_LIST) * NR_EXPERIMENTS_WITH_EACH_CLUSTER + 1

plot_in_2d_initial_data(x_normalized, y, INDEX_IN_COMBINED_PLOT)
INDEX_IN_COMBINED_PLOT += 1

for nr_clusters in NR_CLUSTERS_LIST:
    for experiment in range(NR_EXPERIMENTS_WITH_EACH_CLUSTER):
        kmeans(nr_clusters, plot_after_each_run=False, plot_all_in_one_graph=True)

plt.show()
