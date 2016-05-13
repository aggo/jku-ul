#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 9

# Python:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA


def plotImg(x, index, title):
    x = x[index]
    fig, ax = plt.subplots(figsize=(4, 4))
    x = x.reshape((3, 32, 32))
    x = np.rollaxis(x, 0, 3)
    x = (x - x.min()) / x.ptp()
    img = ax.imshow(x, interpolation="none")
    ax.grid(False)
    ax.axis("off")
    fig.subplots_adjust(left=0, top=1, bottom = 0, right=1)
    plt.title(title)
    # plt.show()
    fig.savefig(title + str(index) + ".png")

data = np.load("cifar10.npy")
plotImg(data, 0, title = "Normal")

# apply pca
pca = PCA(n_components=100)
transformed_data_pca = pca.fit_transform(data)
principal_components_pca = pca.components_
reconstructed_data_pca = np.dot(transformed_data_pca,principal_components_pca)

# apply ICA
ica = FastICA(n_components=100)
transformed_data_ica = ica.fit_transform(data)
principal_components_ica = ica.components_
reconstructed_data_ica = np.dot(transformed_data_ica, principal_components_ica)

# plot images, reconstructed images and components
for i in range(50):
    plotImg(data, i, title = "Normal")
    plotImg(reconstructed_data_pca, i, title = "PCA")
    plotImg(reconstructed_data_ica, i, title = "ICA")

for i in range(100):
    plotImg(principal_components_pca, i, title="PCA-components")
    plotImg(principal_components_ica, i, title="ICA-components")

"""
Comments:
1) How does PCA differ from ICA?
  By looking at the plotted weights, one can notice that PCA tends to produce more and
  more complex-looking components (each more complex than the previous), while those of
  ICA seem unrelated to each other in any way. This is due to the fact that, while both
  look for vectors that explain the data, PCA finds them in such a way as to explain the
  variability in the images (increasingly complex components needed to account for whatever
  amount of variance the previous component didn't capture) and ICA finds the vectors such
  that when projected to them, the images are statistically independent (hence their
  randomness towards each other).

2) PCA's  components are related to each other in the sense that comp. i+1 tries
  to explain the best amount of variance that component i didn't manage to explain.
  (in an orthogonal direction).
  The difference in the types of information the first components extract compared
  to the last ones is due to the decrease in possible directions that can be considered
  when finding the next most variant direction, resuting in more and more complex patterns.

3) The first few components and the last few in ICA do not depend in any way on each other,
  this making them look unrelated. Each of them tries to extract the same thing: a direction
  in which the data is statistically independent, not influencing each other in any way like
  in the case of PCA when the amount of variance one can find depends on what has been found so far.
"""