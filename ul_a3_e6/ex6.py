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
            row = [float(i) for i in row]
            instances.append(row)
    size = len(instances)
    return instances, size

def center_data(data):
    data = data - np.mean(data, axis=0)
    return data

def plotImg(x, title):
    from matplotlib import cm
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.pcolor(x.reshape(45, 50).T, cmap=cm.gray)
    ax.set_ylim([45, 0])
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(title+'.png', bbox_inches='tight')

def get_number_of_components_to_keep_by_desired_variance(desired_variance_retention_percentage,
                                                         retained_variance_percentages):
    sum = 0
    components_to_retain = 0
    index = 0
    while sum + retained_variance_percentages[index] <= desired_variance_retention_percentage:
        sum += retained_variance_percentages[index]
        components_to_retain += 1
        index += 1
    return components_to_retain


def save_original_images_to_png(data, image_indices):
    for image_index in image_indices:
        image = data[image_index, :]
        name = "Original_"+str(image_index)+".png"
        plotImg(image, name)


if __name__ == "__main__":
    image_indices = [0, 1, 3, 10, 12, 23]
    desired_variance_retention_percentages = [0.50, 0.75, 0.90, 0.95, 0.99]

    import numpy as np, matplotlib.pyplot as plt

    # read the data from 1st file
    data, size = read_data('faces94.csv')
    data = np.array(data)
    save_original_images_to_png(data, image_indices)

    data = center_data(np.array(data))

    # downproject without a given number of components to get the explained_variance_ratio_
    from sklearn.decomposition import PCA
    pca_for_faces = PCA()
    pca_for_faces.fit(data)

    # how much variance does each principal component preserve
    retained_variance_percentages = pca_for_faces.explained_variance_ratio_

    # experiments: for each desired variance, compute how many principal components are needed
    # and then save reconstructions of the images.
    for variance_percentage in desired_variance_retention_percentages:
        nr_kept_components = get_number_of_components_to_keep_by_desired_variance(variance_percentage,
                                                                                  retained_variance_percentages)
        fitted_pca = PCA(nr_kept_components).fit(data)

        # the principal components are some "masks"/"eigenfaces" that manage
        # to help distinguish faces one from another
        # (inspired by https://www.youtube.com/watch?v=_lY74pXWlS8)
        principal_components = fitted_pca.components_
        for index in range(len(principal_components)):
            component = principal_components[index]

        # project the data on the previously computed principal components
        data_transformed = fitted_pca.transform(data)

        # in order to reconstruct the images, we use the "eigenfaces", each of them
        # having a certain "weight" in the final image (=> a linear comb. of engenfaces
        # and weights. The "weights" are given by the transformed data)
        for image_index in image_indices:
            reconstructed_image1 = np.dot(data_transformed[image_index,:],principal_components)
            title = "Reconstructed_"+str(image_index)+"_atVar_"+str(variance_percentage)
            plotImg(reconstructed_image1, title=title)
