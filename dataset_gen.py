import math
import os

import matplotlib.pyplot as plt
import scipy.signal
from scipy.fft import fft, fftfreq
import numpy as np


def sine_wave(frequency=440.0, samplerate=88200, t=5000):
    t = np.arange(t)
    sinusoid = np.sin(2 * np.pi * t * (frequency / samplerate))
    return sinusoid


VISUALISE_NORMAL = True
VISUALISE_ABNORMAL = True

#176001
for frequency in range(440, 8801, 440):
    samplerate = 44100

    # Normal wave
    y = [j for j in sine_wave(frequency=frequency, samplerate=samplerate)]
    x = range(0, len(y))

    # Do the FFT
    values = fft(y)
    frequencies = fftfreq(len(x), 1 / samplerate)

    if VISUALISE_NORMAL:
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

    os.makedirs("./dataset/normal", exist_ok=True)
    np.save(f"./dataset/normal/{frequency}", np.array([len(y), y]))

    # Abnormal wave
    t = np.linspace(0, 5, int(1000 * 5))
    y2 = scipy.signal.chirp(t, frequency, 5, frequency + 880, method="quadratic")
    x2 = range(0, len(y2))

    # Do the FFT
    values = fft(y2)
    frequencies = fftfreq(len(x2), 1 / samplerate)

    if VISUALISE_ABNORMAL:
        # Show the frequency plot for each channel
        plt.subplot(211)
        plt.title(f"Frequency: {frequency}")
        plt.plot(frequencies, np.abs(values))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Value")

        plt.subplot(212)
        powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(y2, Fs=samplerate)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.colorbar()
        plt.show()

    os.makedirs("./dataset/abnormal", exist_ok=True)
    np.save(f"./dataset/abnormal/{frequency}", np.array([len(y2), y2]))
