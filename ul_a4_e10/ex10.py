#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 10

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

def combine_songs_and_apply_mixing_matrix(song1, song2):
    # clip longer song
    min = len(song2)
    if (len(song1)<len(song2)):
        min = len(song1)

    song2 = song2[:min]
    song1 = song1[:min]

    combined_songs = np.c_[song1, song2] # as seen in http://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html

    # matrix with det <> 0 (hopefully)
    # mixing_matrix = np.random.rand(2,2)
    # matrix with 0 det => unseparable sources
    mixing_matrix = np.mat([[1,1],[2,2]])

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


freq1, song1 = process_one_song("anthem1.wav")
freq1, song2 = process_one_song("anthem2.wav")

mixed_songs = combine_songs_and_apply_mixing_matrix(song1, song2)
write_to_wav_file(mixed_songs, freq1, "mixed")

scaled_mixed = scale_signal(mixed_songs)
write_to_wav_file(scaled_mixed, freq1, "scaled")

# apply PCA
pca = PCA(n_components=2)
reconstructed_songs_pca = pca.fit_transform(mixed_songs) # reconstruct songs

reconstructed_songs_pca = scale_signal(reconstructed_songs_pca)
write_to_wav_file(reconstructed_songs_pca[:,0], freq1, "song1PCA")
write_to_wav_file(reconstructed_songs_pca[:,1], freq1, "song2PCA")


# apply ICA
ica = FastICA(n_components=2)
reconstructed_songs_ica = ica.fit_transform(mixed_songs)
write_to_wav_file(reconstructed_songs_ica[:,0], freq1, "song1ICA")
write_to_wav_file(reconstructed_songs_ica[:,1], freq1, "song2ICA")

"""
ICA works better at separating the mixture. However, when the matrix we use for mixing
has a determinant = 0, then ICA does not converge, since this matrix does not have an
inverse (needed for ecuations in slide 2/10). Hence it cannot separate the sources.
This can be noticed in the files in the folder 1111matrix where a matrix of all
ones was used.

"""