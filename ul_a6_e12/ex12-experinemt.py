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

##################################################################################################
def read_file(filename):
    csvdata = np.genfromtxt(filename, delimiter=',')
    if len(csvdata.shape)>1:
        csvdata = csvdata[1:, :]  # remove the first line, the header
    else:
        csvdata = csvdata[1:]
    return csvdata

Factor_loading_matrix = np.array([0.5, 0.5, -10, -1, 1, 2, 10, -500]).reshape(8,1)


NR_SAMPLES, NR_FEATURES, NR_FACTORS = 1, Factor_loading_matrix.shape[0], Factor_loading_matrix.shape[1]

Covariance_of_noise = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
Covariance_of_noise = np.diag(Covariance_of_noise) # convert to diagonal matrix
Covariance_of_factors = np.eye(NR_FACTORS)

Mean_of_noise = np.zeros(Covariance_of_noise.shape[0])
Mean_of_factors = np.zeros(NR_FACTORS)

# data normalization: data_normal = preprocessing.scale(data) # Normalization
Noise = Noise = np.random.multivariate_normal(Mean_of_noise, Covariance_of_noise, size=NR_SAMPLES)
Factors = np.random.multivariate_normal(Mean_of_factors, Covariance_of_factors, size=NR_SAMPLES)

Samples = Factors.dot((Factor_loading_matrix.T))+Noise

from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=1)
fa.fit(Samples)

import pandas as pd
print(pd.DataFrame(fa.components_.T))
print(pd.DataFrame(Factor_loading_matrix))
print(pd.DataFrame(Factor_loading_matrix-fa.components_.T))
