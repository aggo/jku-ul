#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 14

import numpy as np
from sklearn.preprocessing import scale


def read_file(filename):
    csvdata = np.genfromtxt(filename, delimiter=',')
    if len(csvdata.shape)>1:
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
    points_location = np.random.rand(nr_clusters,2)  # we have 2D data
    centroids_map = {}
    for i in range(nr_clusters):
        centroids_map[i]=points_location[i]
    return centroids_map

def build_update_map_for_centroids(nr_clusters):
    update_map = {}
    for i in range(nr_clusters):
        update_map[i]= [0, 0, 0] # 3 values: 1st for the sum of X1s, 2nd for the sum of X2s, 3rd for # of terms in sum
        # the new coordinates will be be computed by dividing the sums to the number
    return update_map

def build_centroids_to_assigned_points_map(nr_clusters):
    centroids_to_assigned_points_map = {}
    for i in range(nr_clusters):
        centroids_to_assigned_points_map[i]= [] # 3 values: 1st for the sum of X1s, 2nd for the sum of X2s, 3rd for # of terms in sum
        # the new coordinates will be be computed by dividing the sums to the number
    return centroids_to_assigned_points_map

def plot_in_2d(points, centroids, title):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(points[:,0], points[:,1], 'b.', [i[0] for i in centroids], [i[1] for i in centroids], 'ro')
    plt.ylim([-3,3])
    plt.xlim([-3,3])
    plt.xlabel("X1 coordinate")
    plt.ylabel("X2 coordinate")
    plt.title(title)
    plt.show()

def kmeans(nr_clusters):
    # pick nr_clusters centroids at random - between 0 and 1 since we normalized the data
    centroids = build_centroids_map(nr_clusters)
    update_map_for_centroids = build_update_map_for_centroids(nr_clusters)
    centroids_to_assigned_points_map = build_centroids_to_assigned_points_map(nr_clusters)

    while(True):
        for point in x_normalized:
            min_dist = 1000000;
            chosen_centroid = None
            for centroid in centroids.iteritems():
                current_dist = computeDistance(centroid[1], point)
                if current_dist<min_dist:
                    chosen_centroid = centroid[0]
                    min_dist = current_dist

            update_map_for_centroids[chosen_centroid][0]+=point[0]
            update_map_for_centroids[chosen_centroid][1]+=point[1]
            update_map_for_centroids[chosen_centroid][2]+=1
            centroids_to_assigned_points_map[chosen_centroid].append(point)

        # compute new centroids centers
        for centroid in centroids.iteritems():
            update_map_current_centroid = update_map_for_centroids[centroid[0]]
            if update_map_current_centroid[2]!=0: # if current centroid has no points assigned
                centroid[1][0] = update_map_current_centroid[0]/update_map_current_centroid[2]
                centroid[1][1] = update_map_current_centroid[1]/update_map_current_centroid[2]
            else:
                centroid[1][0] = 0
                centroid[1][1] = 0

            update_map_for_centroids[centroid[0]] = [0,0,0] # reinitialize the map for the next round of point distance comp

        plot_in_2d(x_normalized, [i for i in centroids.itervalues()], "Points and centroids")

data = read_file("kmeansdata.csv")
x = data[:,:-1]
x_normalized = scale(x)
y = data[:,-1]
kmeans(nr_clusters=3)
