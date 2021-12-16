# Analysis Imports
import math
import numpy as np
from scipy.signal import detrend
# Logistical Imports
import warnings
import timeit
from functools import partial
from multiprocessing import Pool, cpu_count
# Visualization imports
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.signal import chirp
from matplotlib.ticker import LogLocator, FixedLocator, MaxNLocator, Formatter, ScalarFormatter
from scipy.signal.windows import dpss


def get_dpss(data, win_size, fs, bandwidth=5, num_tapers=40, nfft=1024, weighting=None):
    tapers, dps = dpss(2000, bandwidth, num_tapers, return_ratios=True)
    tapered_data = np.multiply(np.mat(data).T, np.mat(tapers.T))
    wt = np.ones(num_tapers) / num_tapers
    wt = np.reshape(wt, (num_tapers, 1))  # reshape as column vector
    mt_spectrum = np.dot(tapered_data, wt)
   
    return mt_spectrum



if __name__ == "__main__":
    print("started")
    sz = pd.read_csv("/Users/niklashjort/Desktop/Notes/Speciale/projects/DataHandling/Køge/foo.csv", index_col=False, header=None)
    sz.head()
    series = sz[0]
    series = np.array(series)

    fs = 500       # Define the sampling frequency,
    interval = int(fs/4)   # ... the interval size,
    overlap = int(interval*0.99)  # ... and the overlap intervals

    x = get_dpss(data=series, win_size=500, fs=500)
    print(type(x))
    x = np.asfarray(x)
    f, t, Sxx = scipy.signal.spectrogram(series, fs, nperseg=interval, noverlap=overlap, nfft=1024)
    # option 1: remove some entries
    mask = (f < 40) | (f > 55)
    Sxx = Sxx[mask,:]
    f = f[mask]
    mask = (f < 97) | (f > 103)
    Sxx = Sxx[mask,:]
    f = f[mask]
    


    plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='auto', cmap='jet')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    widths = np.arange(1, 31)

    cwtmatr = scipy.signal.cwt(series, scipy.signal.ricker, widths)
    plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.show()


    

    

    
