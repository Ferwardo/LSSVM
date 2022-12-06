import math

import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np


def sine_wave(frequency=440.0, samplerate=88200, t=5000):
    t = np.arange(t)
    sinusoid = np.sin(2 * np.pi * t * (frequency / samplerate))
    return sinusoid


for frequency in range(440, 4401, 440):
    samplerate = 44100
    y = [j for j in sine_wave(frequency=frequency, samplerate=samplerate)]
    x = range(0, len(y))

    # Do the FFT
    values = fft(y)
    frequencies = fftfreq(len(x), 1 / samplerate)

    # Show the frequency plot for each channel
    plt.subplot(211)
    plt.title(f"Frequency: {frequency}")
    plt.plot(frequencies, np.abs(values))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Value")

    plt.subplot(212)
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(y, Fs=samplerate)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar()
    plt.show()

    np.save(f"./dataset/sine/{frequency}", np.array([len(y), y]))
