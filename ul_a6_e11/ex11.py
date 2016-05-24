#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 13

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
def print_length(songname, song, frequency):
    print("Song {0}'s length is {1}".format(songname, song.shape[0] / frequency))


def play_song(songname):
    # playing the song
    p = pyaudio.PyAudio()

    song = wave.open(songname, 'rb')
    chunk = 1024
    stream = p.open(format=
                    p.get_format_from_width(song.getsampwidth()),
                    channels=song.getnchannels(),
                    rate=song.getframerate(),
                    output=True)

    # read data (based on the chunk size)
    data = song.readframes(chunk)
    # play stream (looping from beginning of file to the end)
    while data != '':
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = song.readframes(chunk)
        print(data)

    # cleanup stuff.
    stream.close()
    p.terminate()

def plot_tone(left_channel, frequency):
    import matplotlib.pyplot as plt
    nr_sample_points = left_channel.shape[0]
    time_array = np.arange(0, nr_sample_points, 1)
    time_array = time_array/frequency
    time_array = time_array * 1000 # scale to miliseconds

    plt.plot(time_array, left_channel, color='k')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (ms)')
    plt.show()
##################################################################################################	


def process_one_song(songname):
    frequency, song = wavfile.read(songname)
    pprint(song)
    # play_song(songname)
    print_length(songname, song, frequency)

    # extract left channel if stereo song
    if len(song.shape)==2:
        left_channel = song[:,0]
    else:
        left_channel = song

    # plot the tone
    # plot_tone(left_channel, frequency)
    return frequency, left_channel

def combine_songs_and_apply_mixing_matrix(song1, song2, song3, mixing_matrix):
    # clip longer song
    min = len(song2)
    for song in [song1, song2, song3]:
        if (len(song))<min:
            min = len(song)

    song3 = song3[:min]
    song2 = song2[:min]
    song1 = song1[:min]

    combined_songs = np.c_[song1, song2, song3] # as seen in http://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html

    mixed = np.dot(combined_songs, mixing_matrix.T)
    return mixed

def scale_signal(signal):
    maxi = np.max(signal)
    mini = np.min(signal)
    a = -1
    b = 1
    scaled_sig = ((b-a) * (signal - mini)) / (maxi - mini) + a
    return scaled_sig


def write_to_wav_file(scaled_mixed, rate, title):
    wavfile.write(title+".wav",rate,data=scaled_mixed)
    # play_song("scaled_mixed.wav")

freq1, song1 = process_one_song("imperial1.wav")
freq2, song2 = process_one_song("rebel-theme2.wav")
freq3, song3 = process_one_song("starwars3.wav")
freq4, song4 = process_one_song("jabba4.wav")
freq5, song5 = process_one_song("r2d2-5.wav")
freq6, song6 = process_one_song("yoda6.wav")

# matrix with det <> 0 (hopefully)
# mixing_matrix = np.random.rand(2,2)
# matrix with 0 det => unseparable sources
mixing_matrix1 = np.mat([[1, 1, 1 ], [2, 2, 2], [3,3,3]])
mixing_matrix2 = np.mat([[1,2,3],[4,5,6],[7,8,9]])
mixing_matrix3 = np.mat([[1,0,0], [0.01, 0,0],[0.01,0,0]])

index = 1
for mixing_matrix in [mixing_matrix1, mixing_matrix2, mixing_matrix3]:
    mixed_songs1 = combine_songs_and_apply_mixing_matrix(song1, song2, song3, mixing_matrix)
    scaled_mixed1 = scale_signal(mixed_songs1)
    write_to_wav_file(scaled_mixed1, freq1, "mixed123-using-mat"+str(index))

    mixed_songs2 = combine_songs_and_apply_mixing_matrix(song4, song5, song6, mixing_matrix)
    scaled_mixed2 = scale_signal(mixed_songs2)
    write_to_wav_file(scaled_mixed2, freq4, "mixed345--using-mat"+str(index))

    # apply ICA
    ica = FastICA(n_components=3)
    reconstructed_songs123_ica = ica.fit_transform(mixed_songs1)
    write_to_wav_file(reconstructed_songs123_ica[:,0], freq1, "song1ICA-using-mat"+str(index))
    write_to_wav_file(reconstructed_songs123_ica[:,1], freq1, "song2ICA-using-mat"+str(index))
    write_to_wav_file(reconstructed_songs123_ica[:,2], freq1, "song3ICA-using-mat"+str(index))

    ica = FastICA(n_components=3)
    reconstructed_songs456_ica = ica.fit_transform(mixed_songs2)
    write_to_wav_file(reconstructed_songs456_ica[:, 0], freq4, "song4ICA-using-mat"+str(index))
    write_to_wav_file(reconstructed_songs456_ica[:, 1], freq4, "song5ICA-using-mat"+str(index))
    write_to_wav_file(reconstructed_songs456_ica[:, 2], freq4, "song6ICA-using-mat"+str(index))

    # apply ICA to extract less signals than sources
    ica = FastICA(n_components=2)
    reconstructed_songs123_ica = ica.fit_transform(mixed_songs1)
    write_to_wav_file(reconstructed_songs123_ica[:, 0], freq1, "song1ICA-less-signals-using-mat" + str(index))
    write_to_wav_file(reconstructed_songs123_ica[:, 1], freq1, "song2ICA-less-signals--using-mat" + str(index))

    ica = FastICA(n_components=2)
    reconstructed_songs456_ica = ica.fit_transform(mixed_songs2)
    write_to_wav_file(reconstructed_songs456_ica[:, 0], freq4, "song4ICA-less-signals--using-mat" + str(index))
    write_to_wav_file(reconstructed_songs456_ica[:, 1], freq4, "song5ICA-less-signals--using-mat" + str(index))
    index += 1

"""
ICA works better at separating the mixture. However, when the matrix we use for mixing
has a determinant = 0, then ICA does not converge, since this matrix does not have an
inverse (needed for ecuations in slide 2/10). Hence it cannot separate the sources.
This can be noticed in the files in the folder 1111matrix where a matrix of all
ones was used.

"""