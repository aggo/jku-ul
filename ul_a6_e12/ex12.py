#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 12

from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from scipy.io import wavfile
import pyaudio
import wave
import sys
def check_up_noise(Noise):
    import pandas as pd
    print(pd.DataFrame(Noise))
    import matplotlib.pyplot as plt
    plt.contour(Noise)
    plt.show()
##################################################################################################
def read_file(filename):
    csvdata = np.genfromtxt(filename, delimiter=',')
    if len(csvdata.shape)>1:
        csvdata = csvdata[1:, :]  # remove the first line, the header
    else:
        csvdata = csvdata[1:]
    return csvdata

Factor_loading_matrix = read_file("U.csv")

NR_SAMPLES, NR_FEATURES, NR_FACTORS = 1000, Factor_loading_matrix.shape[0], Factor_loading_matrix.shape[1]

Covariance_of_noise = read_file("Psi.csv")
Covariance_of_noise = np.diag(Covariance_of_noise) # convert to diagonal matrix
Covariance_of_factors = np.eye(NR_FACTORS)

Mean_of_noise = np.zeros(Covariance_of_noise.shape[0])
Mean_of_factors = np.zeros(NR_FACTORS)

# data normalization: data_normal = preprocessing.scale(data) # Normalization
Noise = np.random.multivariate_normal(Mean_of_noise, Covariance_of_noise, size=NR_SAMPLES)

from matplotlib import pyplot as plt

# check_up_noise(Noise)

Factors = np.random.multivariate_normal(Mean_of_factors, Covariance_of_factors, size=NR_SAMPLES)
# Factors = np.random.multivariate_normal(Mean_of_factors, Covariance_of_factors, )

Samples = Factors.dot((Factor_loading_matrix.T))+Noise

from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=5)
fa.fit(Samples)
fa.get_precision()

import pandas as pd
print(pd.DataFrame(fa.components_.T,columns=["V1","V2","V3","V4","V5"]))
z_min, z_max = -np.abs(fa.components_.T).max(), np.abs(fa.components_.T).max()
plt.pcolor(fa.components_.T, cmap='RdBu', vmin=z_min, vmax=z_max)
plt.colorbar()
plt.show()

print(pd.DataFrame(Factor_loading_matrix,columns=["V1","V2","V3","V4","V5"]))

z_min, z_max = -np.abs(Factor_loading_matrix).max(), np.abs(Factor_loading_matrix).max()
plt.pcolor(Factor_loading_matrix, cmap='RdBu', vmin=z_min, vmax=z_max)
plt.colorbar()
plt.show()
