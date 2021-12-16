# Analysis Imports
import math
import numpy as np
from scipy.signal.windows import dpss
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



# MULTITAPER SPECTROGRAM #
def multitaper_spectrogram(data, fs, frequency_range=None, time_bandwidth=5, num_tapers=None, window_params=None,
                           min_nfft=0, detrend_opt='linear', multiprocess=False, cpus=False, weighting='unity',
                           plot_on=True, clim_scale = True, verbose=True, xyflip=False):
    """ Compute multitaper spectrogram of timeseries data
    Usage:
    mt_spectrogram, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range=None, time_bandwidth=5,
                                                                   num_tapers=None, window_params=None, min_nfft=0,
                                                                   detrend_opt='linear', multiprocess=False, cpus=False,
                                                                    weighting='unity', plot_on=True, verbose=True,
                                                                    xyflip=False):
        Arguments:
                data (1d np.array): time series data -- required
                fs (float): sampling frequency in Hz  -- required
                frequency_range (list): 1x2 list - [<min frequency>, <max frequency>] (default: [0 nyquist])
                time_bandwidth (float): time-half bandwidth product (window duration*half bandwidth of main lobe)
                                        (default: 5 Hz*s)
                num_tapers (int): number of DPSS tapers to use (default: [will be computed
                                  as floor(2*time_bandwidth - 1)])
                window_params (list): 1x2 list - [window size (seconds), step size (seconds)] (default: [5 1])
                detrend_opt (string): detrend data window ('linear' (default), 'constant', 'off')
                                      (Default: 'linear')
                min_nfft (int): minimum allowable NFFT size, adds zero padding for interpolation (closest 2^x)
                                (default: 0)
                multiprocess (bool): Use multiprocessing to compute multitaper spectrogram (default: False)
                cpus (int): Number of cpus to use if multiprocess = True (default: False). Note: if default is left
                            as False and multiprocess = True, the number of cpus used for multiprocessing will be
                            all available - 1.
                weighting (str): weighting of tapers ('unity' (default), 'eigen', 'adapt');
                plot_on (bool): plot results (default: True)
                clim_scale (bool): automatically scale the colormap on the plotted spectrogram (default: true)
                verbose (bool): display spectrogram properties (default: true)
                xyflip (bool): transpose the mt_spectrogram output (default: false)
        Returns:
                mt_spectrogram (TxF np array): spectral power matrix
                stimes (1xT np array): timepoints (s) in mt_spectrogram
                sfreqs (1xF np array)L frequency values (Hz) in mt_spectrogram
        Example:
        In this example we create some chirp data and run the multitaper spectrogram on it.
            import numpy as np  # import numpy
            from scipy.signal import chirp  # import chirp generation function
            # Set spectrogram params
            fs = 200  # Sampling Frequency
            frequency_range = [0, 25]  # Limit frequencies from 0 to 25 Hz
            time_bandwidth = 3  # Set time-half bandwidth
            num_tapers = 5  # Set number of tapers (optimal is time_bandwidth*2 - 1)
            window_params = [4, 1]  # Window size is 4s with step size of 1s
            min_nfft = 0  # No minimum nfft
            detrend_opt = 'constant'  # detrend each window by subtracting the average
            multiprocess = True  # use multiprocessing
            cpus = 3  # use 3 cores in multiprocessing
            weighting = 'unity'  # weight each taper at 1
            plot_on = True  # plot spectrogram
            clim_scale = False # don't auto-scale the colormap
            verbose = True  # print extra info
            xyflip = False  # do not transpose spect output matrix
            # Generate sample chirp data
            t = np.arange(1/fs, 600, 1/fs)  # Create 10 min time array from 1/fs to 600 stepping by 1/fs
            f_start = 1  # Set chirp freq range min (Hz)
            f_end = 20  # Set chirp freq range max (Hz)
            data = chirp(t, f_start, t[-1], f_end, 'logarithmic')
            # Compute the multitaper spectrogram
            spect, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                           window_params, min_nfft, detrend_opt, multiprocess,
                                                           cpus, weighting, plot_on, verbose, xyflip):
        This code is companion to the paper:
        "Sleep Neurophysiological Dynamics Through the Lens of Multitaper Spectral Analysis"
           Michael J. Prerau, Ritchie E. Brown, Matt T. Bianchi, Jeffrey M. Ellenbogen, Patrick L. Purdon
           December 7, 2016 : 60-92
           DOI: 10.1152/physiol.00062.2015
         which should be cited for academic use of this code.
         A full tutorial on the multitaper spectrogram can be found at:  #   http://www.sleepEEG.org/multitaper
        Copyright 2021 Michael J. Prerau Laboratory. - http://www.sleepEEG.org
        Authors: Michael J. Prerau, Ph.D., Thomas Possidente
        
        Last modified - 2/18/2021 Thomas Possidente
  __________________________________________________________________________________________________________________
    """

    #  Process user input
    [data, fs, frequency_range, time_bandwidth, num_tapers,
     winsize_samples, winstep_samples, window_start,
     num_windows, nfft, detrend_opt, plot_on, verbose] = process_input(data, fs, frequency_range, time_bandwidth,
                                                                       num_tapers, window_params, min_nfft,
                                                                       detrend_opt, plot_on, verbose)

    # Set up spectrogram parameters
    [window_idxs, stimes, sfreqs, freq_inds] = process_spectrogram_params(fs, nfft, frequency_range, window_start,
                                                                          winsize_samples)
    # Display spectrogram parameters
    if verbose:
        display_spectrogram_props(fs, time_bandwidth, num_tapers, [winsize_samples, winstep_samples], frequency_range,
                                  detrend_opt)

    # Split data into segments and preallocate
    data_segments = data[window_idxs]

    # COMPUTE THE MULTITAPER SPECTROGRAM
    #     STEP 1: Compute DPSS tapers based on desired spectral properties
    #     STEP 2: Multiply the data segment by the DPSS Tapers
    #     STEP 3: Compute the spectrum for each tapered segment
    #     STEP 4: Take the mean of the tapered spectra

    # Compute DPSS tapers (STEP 1)
    dpss_tapers, dpss_eigen = dpss(winsize_samples, time_bandwidth, num_tapers, return_ratios=True)
    dpss_eigen = np.reshape(dpss_eigen, (num_tapers, 1))

    # pre-compute weights
    if weighting == 'eigen':
        wt = dpss_eigen / num_tapers
    elif weighting == 'unity':
        wt = np.ones(num_tapers) / num_tapers
        wt = np.reshape(wt, (num_tapers, 1))  # reshape as column vector
    else:
        wt = 0

    tic = timeit.default_timer()  # start timer

    # set all but 1 arg of calc_mts_segment to constant (so we only have to supply one argument later)
    calc_mts_segment_plus_args = partial(calc_mts_segment, dpss_tapers=dpss_tapers, nfft=nfft, freq_inds=freq_inds,
                                         detrend_opt=detrend_opt, num_tapers=num_tapers, dpss_eigen=dpss_eigen,
                                         weighting=weighting, wt=wt)

    if multiprocess:  # use multiprocessing
        if not cpus:  # if cpus not specfied, use all but 1
            pool = Pool(cpu_count()-1)
        else:  # else us specified number
            pool = Pool(cpus)

        # Compute multiprocess multitaper spect.
        mt_spectrogram = pool.map(calc_mts_segment_plus_args, data_segments)
        pool.close()
        pool.join()

    else:  # if no multiprocessing, compute normally
        mt_spectrogram = np.apply_along_axis(calc_mts_segment_plus_args, 1, data_segments)

    # Compute one-sided PSD spectrum
    mt_spectrogram = np.asarray(mt_spectrogram)
    mt_spectrogram = mt_spectrogram.T
    dc_select = np.where(sfreqs == 0)
    nyquist_select = np.where(sfreqs == fs/2)
    select = np.setdiff1d(np.arange(0, len(sfreqs)), [dc_select, nyquist_select])

    mt_spectrogram = np.vstack([mt_spectrogram[dc_select[0], :], 2*mt_spectrogram[select, :],
                               mt_spectrogram[nyquist_select[0], :]]) / fs

    # Flip if requested
    if xyflip:
        mt_spectrogram = np.transpose(mt_spectrogram)

    # End timer and get elapsed compute time
    toc = timeit.default_timer()
    elapsed_time = toc - tic
    if verbose:
        print("\n Multitaper compute time: " + str(elapsed_time) + " seconds")

    # Plot multitaper spectrogram
    if plot_on:
        pass
        # # Eliminate bad data from colormap scaling
        # spect_data = mt_spectrogram
        # clim = np.percentile(spect_data, [5, 95])  # Scale colormap from 5th percentile to 95th

        # plt.figure(1, figsize=(10, 5))
        # librosa.display.specshow(nanpow2db(mt_spectrogram), x_axis='time', y_axis='linear',
        #                          x_coords=stimes, y_coords=sfreqs, shading='auto', cmap="jet")
        # plt.colorbar(label='Power (dB)')
        # plt.xlabel("Time (HH:MM:SS)")
        # plt.ylabel("Frequency (Hz)")
        # if clim_scale:
        #     plt.clim(clim)  # actually change colorbar scale
        # plt.show()

    # Put outputs into better format for output
    #stimes = np.mat(stimes)
    #sfreqs = np.mat(sfreqs)

    # if all(mt_spectrogram.flatten() == 0):
    #     print("\n Data was all zeros, no output")

    return mt_spectrogram, stimes, sfreqs


# Helper Functions #

# Process User Inputs #
def process_input(data, fs, frequency_range=None, time_bandwidth=5, num_tapers=None, window_params=None, min_nfft=0,
                  detrend_opt='linear', plot_on=True, verbose=True):
    # Make sure data is 1 dimensional np array
    if len(data.shape) != 1:
        if (len(data.shape) == 2) & (data.shape[1] == 1):  # if it's 2d, but can be transferred to 1d, do so
            data = np.ravel(data[:, 0])
        elif (len(data.shape) == 2) & (data.shape[0] == 1):  # if it's 2d, but can be transferred to 1d, do so
            data = np.ravel(data.T[:, 0])
        else:
            raise TypeError("Input data is the incorrect dimensions. Should be a 1d array with shape (n,) where n is \
                            the number of data points. Instead data shape was " + str(data.shape))

    # Set frequency range if not provided
    if frequency_range is None:
        frequency_range = [0, fs / 2]

    # Set detrending method
    detrend_opt = detrend_opt.lower()
    if detrend_opt != 'linear':
        if detrend_opt in ['const', 'constant']:
            detrend_opt = 'constant'
        elif detrend_opt in ['none', 'false']:
            detrend_opt = 'off'
        else:
            raise ValueError("'" + str(detrend_opt) + "' is not a valid argument for detrend_opt. The choices " +
                             "are: 'constant', 'linear', or 'off'.")
    # Check if frequency range is valid
    if frequency_range[1] > fs / 2:
        frequency_range[1] = fs / 2
        warnings.warn('Upper frequency range greater than Nyquist, setting range to [' +
                      str(frequency_range[0]) + ', ' + str(frequency_range[1]) + ']')

    # Set number of tapers if none provided
    if num_tapers is None:
        num_tapers = math.floor(2 * time_bandwidth) - 1

    # Warn if number of tapers is suboptimal
    if num_tapers != math.floor(2 * time_bandwidth) - 1:
        warnings.warn('Number of tapers is optimal at floor(2*TW) - 1. consider using ' +
                      str(math.floor(2 * time_bandwidth) - 1))

    # If no window params provided, set to defaults
    if window_params is None:
        window_params = [5, 1]

    # Check if window size is valid, fix if not
    if window_params[0] * fs % 1 != 0:
        winsize_samples = round(window_params[0] * fs)
        warnings.warn('Window size is not divisible by sampling frequency. Adjusting window size to ' +
                      str(winsize_samples / fs) + ' seconds')
    else:
        winsize_samples = window_params[0] * fs

    # Check if window step is valid, fix if not
    if window_params[1] * fs % 1 != 0:
        winstep_samples = round(window_params[1] * fs)
        warnings.warn('Window step size is not divisible by sampling frequency. Adjusting window step size to ' +
                      str(winstep_samples / fs) + ' seconds')
    else:
        winstep_samples = window_params[1] * fs

    # Get total data length
    len_data = len(data)

    # Check if length of data is smaller than window (bad)
    if len_data < winsize_samples:
        raise ValueError("\nData length (" + str(len_data) + ") is shorter than window size (" +
                         str(winsize_samples) + "). Either increase data length or decrease window size.")

    # Find window start indices and num of windows
    window_start = np.arange(0, len_data - winsize_samples+1, winstep_samples)
    num_windows = len(window_start)

    # Get num points in FFT
    if min_nfft == 0:  # avoid divide by zero error in np.log2(0)
        nfft = max(2 ** math.ceil(np.log2(abs(winsize_samples))), winsize_samples)
    else:
        nfft = max(max(2 ** math.ceil(np.log2(abs(winsize_samples))), winsize_samples),
                   2 ** math.ceil(np.log2(abs(min_nfft))))

    return ([data, fs, frequency_range, time_bandwidth, num_tapers,
             int(winsize_samples), int(winstep_samples), window_start, num_windows, nfft,
             detrend_opt, plot_on, verbose])


# PROCESS THE SPECTROGRAM PARAMETERS #
def process_spectrogram_params(fs, nfft, frequency_range, window_start, datawin_size):

    # create frequency vector
    df = fs / nfft
    sfreqs = np.arange(0, fs, df)

    # Get frequencies for given frequency range
    freq_inds = (sfreqs >= frequency_range[0]) & (sfreqs <= frequency_range[1])
    sfreqs = sfreqs[freq_inds]

    # Compute times in the middle of each spectrum
    window_middle_samples = window_start + round(datawin_size / 2)
    stimes = window_middle_samples / fs

    # Get indexes for each window
    window_idxs = np.atleast_2d(window_start).T + np.arange(0, datawin_size, 1)
    window_idxs = window_idxs.astype(int)

    return [window_idxs, stimes, sfreqs, freq_inds]


# DISPLAY SPECTROGRAM PROPERTIES
def display_spectrogram_props(fs, time_bandwidth, num_tapers, data_window_params, frequency_range, detrend_opt):

    data_window_params = np.asarray(data_window_params) / fs

    # Print spectrogram properties
    print("Multitaper Spectrogram Properties: ")
    print('     Spectral Resolution: ' + str(2 * time_bandwidth / data_window_params[0]) + 'Hz')
    print('     Window Length: ' + str(data_window_params[0]) + 's')
    print('     Window Step: ' + str(data_window_params[1]) + 's')
    print('     Time Half-Bandwidth Product: ' + str(time_bandwidth))
    print('     Number of Tapers: ' + str(num_tapers))
    print('     Frequency Range: ' + str(frequency_range[0]) + "-" + str(frequency_range[1]) + 'Hz')
    print('     Detrend: ' + detrend_opt + '\n')


# NANPOW2DB
def nanpow2db(y):
    if isinstance(y, int) or isinstance(y, float):
        if y == 0:
            return np.nan
        else:
            ydB = 10 * np.log10(y)
    else:
        if isinstance(y, list):  # if list, turn into array
            y = np.asarray(y)
        y = y.astype(float)  # make sure it's a float array so we can put nans in it
        y[y == 0] = np.nan
        ydB = 10 * np.log10(y)

    return ydB


# Helper #
def is_outlier(data):
    smad = 1.4826 * np.median(abs(data - np.median(data)))  # scaled median absolute deviation
    outlier_mask = abs(data-np.median(data)) > 3*smad  # outliers are more than 3 smads away from median
    outlier_mask = (outlier_mask | np.isnan(data) | np.isinf(data))
    return outlier_mask


# CALCULATE MULTITAPER SPECTRUM ON SINGLE SEGMENT
def calc_mts_segment(data_segment, dpss_tapers, nfft, freq_inds, detrend_opt, num_tapers, dpss_eigen, weighting, wt):
    # If segment has all zeros, return vector of zeros
    if all(data_segment == 0):
        ret = np.empty(sum(freq_inds))
        ret.fill(0)
        return ret

    # Option to detrend data to remove low frequency DC component
    if detrend_opt != 'off':
        data_segment = detrend(data_segment, type=detrend_opt)

    # Multiply data by dpss tapers (STEP 2)
    tapered_data = np.multiply(np.mat(data_segment).T, np.mat(dpss_tapers.T))

    # Compute the FFT (STEP 3)
    fft_data = np.fft.fft(tapered_data, nfft, axis=0)

    # Compute the weighted mean spectral power across tapers (STEP 4)
    spower = np.power(np.imag(fft_data), 2) + np.power(np.real(fft_data), 2)
    if weighting == 'adapt':
        # adaptive weights - for colored noise spectrum (Percival & Walden p368-370)
        tpower = np.dot(np.transpose(data_segment), (data_segment/len(data_segment)))
        spower_iter = np.mean(spower[:, 0:2], 1)
        spower_iter = spower_iter[:, np.newaxis]
        a = (1 - dpss_eigen) * tpower
        for i in range(3):  # 3 iterations only
            # Calc the MSE weights
            b = np.dot(spower_iter, np.ones((1, num_tapers))) / ((np.dot(spower_iter, np.transpose(dpss_eigen))) +
                                                                 (np.ones((nfft, 1)) * np.transpose(a)))
            # Calc new spectral estimate
            wk = (b**2) * np.dot(np.ones((nfft, 1)), np.transpose(dpss_eigen))
            spower_iter = np.sum((np.transpose(wk) * np.transpose(spower)), 0) / np.sum(wk, 1)
            spower_iter = spower_iter[:, np.newaxis]

        mt_spectrum = np.squeeze(spower_iter)

    else:
        # eigenvalue or uniform weights
        mt_spectrum = np.dot(spower, wt)
        mt_spectrum = np.reshape(mt_spectrum, nfft)  # reshape to 1D

    return mt_spectrum[freq_inds]

class TimeFormatter(Formatter):
    def __init__(self, lag=False):

        self.lag = lag


    def __call__(self, x, pos=None):
        '''Return the time format as pos'''

        _, dmax = self.axis.get_data_interval()
        vmin, vmax = self.axis.get_view_interval()

        # In lag-time axes, anything greater than dmax / 2 is negative time
        if self.lag and x >= dmax * 0.5:
            # In lag mode, don't tick past the limits of the data
            if x > dmax:
                return ''
            value = np.abs(x - dmax)
            # Do we need to tweak vmin/vmax here?
            sign = '-'
        else:
            value = x
            sign = ''

        if vmax - vmin > 3600:
            s = '{:d}:{:02d}:{:02d}'.format(int(value / 3600.0),
                                            int(np.mod(value / 60.0, 60)),
                                            int(np.mod(value, 60)))
        elif vmax - vmin > 60:
            s = '{:d}:{:02d}'.format(int(value / 60.0),
                                     int(np.mod(value, 60)))
        else:
            s = '{:.2g}'.format(value)

        return '{:s}{:s}'.format(sign, s)


def fft_frequencies(sr=22050, n_fft=2048):
    return np.linspace(0,
                        float(sr) / 2,
                        int(1 + n_fft//2),
                        endpoint=True)

def __coord_fft_hz(n, sr=22050, **_kwargs):
    '''Get the frequencies for FFT bins'''
    n_fft = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = fft_frequencies(sr=sr, n_fft=n_fft)
    fmax = basis[-1]
    basis -= 0.5 * (basis[1] - basis[0])
    basis = np.append(np.maximum(0, basis), [fmax])
    return basis


def samples_to_time(samples, sr=22050):
    return np.asanyarray(samples) / float(sr)

def frames_to_time(frames, sr=22050, hop_length=512, n_fft=None):
    samples = frames_to_samples(frames,
                                hop_length=hop_length,
                                n_fft=n_fft)

    return samples_to_time(samples, sr=sr)

def frames_to_samples(frames, hop_length=512, n_fft=None):
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)
    return (np.asanyarray(frames) * hop_length + offset).astype(int)

def __coord_time(n, sr=22050, hop_length=512, **_kwargs):
    return frames_to_time(np.arange(n+1), sr=sr, hop_length=hop_length)

def __mesh_coords(ax_type, coords, n, **kwargs):
    '''Compute axis coordinates'''

    if coords is not None:
        if len(coords) < n:
            raise ValueError('Coordinate shape mismatch: '
                                 '{}<{}'.format(len(coords), n))
        return coords

    coord_map = {'linear': __coord_fft_hz,
                 'hz': __coord_fft_hz,
                 'log': __coord_fft_hz,}

    if ax_type not in coord_map:
        raise ValueError('Unknown axis type: {}'.format(ax_type))

    return coord_map[ax_type](n, **kwargs)

def __decorate_axis(axis, ax_type):
    if ax_type == 'time':
        axis.set_major_formatter(TimeFormatter(lag=False))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Time')
    elif ax_type in ['linear', 'hz']:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_label_text('Hz')

if __name__ == "__main__":
    print("started")
    sz = pd.read_csv("/Users/niklashjort/Desktop/Notes/Speciale/projects/DataHandling/Køge/foo.csv", index_col=False, header=None)
    sz.head()
    series = sz[0]
    series = np.array(series)
    fs = 500
    frequency_range = [0, 25]  # Limit frequencies from 0 to 25 Hz
    time_bandwidth = 3  # Set time-half bandwidth
    num_tapers = 5  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    window_params = [4, 1]  # Window size is 4s with step size of 1s
    min_nfft = 0  # No minimum nfft
    detrend_opt = 'constant'  # detrend each window by subtracting the average
    multiprocess = True  # use multiprocessing
    cpus = 3  # use 3 cores in multiprocessing
    weighting = 'unity'  # weight each taper at 1
    plot_on = True  # plot spectrogram
    clim_scale = False # do not auto-scale colormap
    verbose = True  # print extra info
    xyflip = False  # do not transpose spect output matrix

    # Generate sample chirp data
    t = np.arange(1/fs, 600, 1/fs)  # Create 10 min time array from 1/fs to 600 stepping by 1/fs
    f_start = 1  # Set chirp freq range min (Hz)
    f_end = 20  # Set chirp freq range max (Hz)
    data = chirp(t, f_start, t[-1], f_end, 'logarithmic')


    Sxx, t, f = multitaper_spectrogram(series, 500, window_params=[1, 0.4], num_tapers=20, min_nfft=1024, frequency_range=[0, 40])
    Sxx = nanpow2db(Sxx)
    x_coords = __mesh_coords('time', t, Sxx.shape[1])
    y_coords = __mesh_coords('linear', f, Sxx.shape[0])
    
    axes = plt.gca()
    axes.set_xlim(x_coords.min(), x_coords.max())
    #axes.set_ylim(y_coords.min(), y_coords.max())

    # Construct tickers and locators
    __decorate_axis(axes.xaxis, 'time')
    #__decorate_axis(axes.yaxis, 'linear')
    out = axes.pcolormesh(x_coords, y_coords, Sxx, cmap='jet', shading='auto')
    #axes.set_xlim(x_coords.min(), x_coords.max())
    #axes.set_ylim(y_coords.min(), y_coords.max())

    # # Construct tickers and locators
    # __decorate_axis(axes.xaxis, 'time')
    # __decorate_axis(axes.yaxis, 'linear')
    plt.sci(out)
    plt.colorbar(label='Power (dB)')
    axes.set_xlim(x_coords.min(), x_coords.max())
    axes.set_ylim(y_coords.min(), y_coords.max())
    # Construct tickers and locators
    __decorate_axis(axes.xaxis, 'time')
    __decorate_axis(axes.yaxis, 'linear')
    # plt.xlabel("Time (HH:MM:SS)")
    # plt.ylabel("Frequency (Hz)")

    # f, t, Sxx = scipy.signal.spectrogram(series, fs)
    # plt.pcolormesh(t, f, 10* np.log10(Sxx), shading='auto', cmap='jet')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    


    # # Set up axis scaling
    # __scale_axes(axes, x_axis, 'x')
    # __scale_axes(axes, y_axis, 'y')

    


    print(type(t))
    print(type(f))
    #plt.pcolormesh(t, f, Sxx, cmap='jet')
    plt.show()