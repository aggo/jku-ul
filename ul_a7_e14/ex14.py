#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 14

import numpy as np
from matplotlib import animation
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

# def plot_in_2d(points, centroids, title):
#
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     plt.plot(points[:,0], points[:,1], 'b.', [i[0] for i in centroids], [i[1] for i in centroids], 'ro')
#     plt.ylim([-3,3])
#     plt.xlim([-3,3])
#     plt.xlabel("X1 coordinate")
#     plt.ylabel("X2 coordinate")
#     plt.title(title)
#     plt.show()


def plot_in_2d_each_cluster(centroids_to_assigned_points_map, centroids, imageOrder):
    ax1.clear()
    for centroid in centroids_to_assigned_points_map.iterkeys():
        points = centroids_to_assigned_points_map[centroid]
        ax1.scatter([i[0] for i in points], [i[1] for i in points],  marker = 'o', color = colors(centroid*100),)
        ax1.scatter(centroids[centroid][0], centroids[centroid][1], marker='o', s=81, linewidths=2,
            color='white', zorder=10, edgecolors="black")
    plt.draw()
    plt.pause(0.05)

    # plt.draw()



def kmeans(nr_clusters, plot_after_each_run=True):
    # pick nr_clusters centroids at random - between 0 and 1 since we normalized the data
    centroids = build_centroids_map(nr_clusters)
    nr_runs = 1

    while(nr_runs<=MAX_ITERATIONS):
        update_map_for_centroids = build_update_map_for_centroids(nr_clusters)
        centroids_to_assigned_points_map = build_centroids_to_assigned_points_map(nr_clusters)
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
        if plot_after_each_run:
            plot_in_2d_each_cluster(centroids_to_assigned_points_map, centroids, nr_runs)
        nr_runs+=1
    plot_in_2d_each_cluster(centroids_to_assigned_points_map, centroids, nr_runs)

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
colors = plt.get_cmap("rainbow")
plt.ylim([-3, 3])
plt.xlim([-3, 3])

MAX_ITERATIONS = 20
data = read_file("kmeansdata.csv")
x = data[:,:-1]
x_normalized = scale(x)
y = data[:,-1]
kmeans(nr_clusters=4, plot_after_each_run = False)

plt.waitforbuttonpress()

