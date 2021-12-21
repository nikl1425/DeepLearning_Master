import numpy as np
from scipy import signal
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import psutil
import gc
import matplotlib.colors as colors
#from util import logging_info_txt
import math
import warnings
from scipy.signal.windows import dpss
from functools import partial
from scipy.signal import detrend
from matplotlib.ticker import LogLocator, FixedLocator, MaxNLocator, Formatter, ScalarFormatter

'''
Custom filters for EEG and ECG
'''

from util import logging_info_txt

def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    i, u = signal.butter(order, [low, high], btype='bandstop')
    y = signal.lfilter(i, u, data)
    return y

def apply_filter(series, FREQ):
    series = butter_bandstop_filter(series, 97, 103, FREQ, order=6)
    series = butter_bandstop_filter(series, 47, 53, FREQ, order=6)
    series = butter_highpass_filter(series, 1, FREQ, order=6)
    return series

def spec_save_to_folder(index, win, channel, patient_state, patient, save_path, plot_title = False, FREQ = 256):
    '''
    plotting and saving spectrogram in comprehension.
    '''
    interval = int(FREQ)   # ... the interval size,
    overlap = int(interval * 0.95)  # ... and the overlap of interval
    series = win

    try:
        series = np.array(series).astype(np.float)
    except Exception as e:
        print(f"error: {e}")
        print(f"patient_state: {patient_state} channel: {channel} index: {index} window: {series}")
    
    if plot_title:
        plt.title(f"{channel} : is_seizure = {patient_state}")

    filt_series = apply_filter(series, FREQ)

    f, t, Sxx = signal.spectrogram(np.array(filt_series), fs=FREQ, noverlap=overlap, nperseg=interval, nfft=256, window='hann')
    Sxx = nanpow2db(Sxx)
    #normalize_color= colors.LogNorm(vmin=np.amin(Sxx), vmax=np.amax(Sxx))      
    #Sxx = 10*np.log10(Sxx) 
    plt.pcolormesh(t, f, Sxx, cmap='jet')
    #plt.specgram(filt_series, cmap='jet', Fs=FREQ, NFFT=interval, noverlap=overlap)
    #plt.axis('off')
    plt.show()
    
    Log_file_path = save_path + "Log.txt"

    #LOGGING:
    logging_info_txt(f"patient: {patient} channel: {channel} FREQ: {FREQ} \n", Log_file_path)

    if patient_state == "seizure":
        plt.savefig(f'{save_path}Seizure/{patient}_{index}_{channel}.png', bbox_inches='tight')
    elif patient_state == "interictal":
        plt.savefig(f'{save_path}Interictal/{patient}_{index}_{channel}.png', bbox_inches='tight')
    elif patient_state == "prei_one":
        plt.savefig(f'{save_path}Preictal/{patient}_{index}_{channel}.png', bbox_inches='tight')

    print(f"SUCCES - patient: {patient} index: {index}")
    del series, f, t, Sxx
    plt.clf()    
    plt.close()
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')
    gc.collect()


def multitaper_spec_save_to_folder(index, win, channel, patient_state, patient, save_path, FREQ = 256, decorate=False):
    '''
    Multi taper spectrogram.
    Uses 5 tapers.
    nfft = 1024.
    Nyquist frequency at 40/2 = 20.
    '''

    interval = int(FREQ)   # ... the interval size,
    series = win[0]
    time_of_observation = win[1]
    plt.figure(figsize=(7,7))
    try:
        series = np.array(series).astype(np.float)
    except Exception as e:
        print(f"error: {e}")
        print(f"patient_state: {patient_state} channel: {channel} index: {index} window: {series}")
    


    filt_series = apply_filter(series, FREQ)
    Sxx, t, f = multitaper_spectrogram(filt_series, 256, window_params=[1, 0.25], num_tapers=2, min_nfft=1024, frequency_range=[0, 40])
    Sxx = nanpow2db(Sxx)

    if decorate:
        plt.title(f"{channel} : is_seizure = {patient_state} : {time_of_observation}")
        x_coords = mesh_coords('time', t, Sxx.shape[1])
        y_coords = mesh_coords('linear', f, Sxx.shape[0])
        axes = plt.gca()
        axes.set_xlim(x_coords.min(), x_coords.max())
        decorate_axis(axes.xaxis, 'time')
        decorate_axis(axes.yaxis, 'linear')
        out = axes.pcolormesh(x_coords, y_coords, Sxx, cmap='jet', shading='auto')
        plt.sci(out)
        plt.colorbar(label='Power (dB)')
        axes.set_xlim(x_coords.min(), x_coords.max())
        axes.set_ylim(y_coords.min(), y_coords.max())
    else:
        plt.pcolormesh(t, f, Sxx, cmap='jet', shading="auto")
        plt.axis('off')
        plt.show()

    time_of_observation = str(time_of_observation).replace(":", "-")
    Log_file_path = save_path + "Log.txt"

    # #LOGGING:
    # logging_info_txt(f"patient: {patient} channel: {channel} time: {time_of_observation} FREQ: {FREQ} \n", Log_file_path)

    

    # if patient_state == "seizure":
    #     plt.savefig(f'{save_path}Seizure/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
    # elif patient_state == "interictal":
    #     plt.savefig(f'{save_path}Interictal/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
    # elif patient_state == "prei_one":
    #     plt.savefig(f'{save_path}Preictal/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')

    print(f"SUCCES - patient: {patient} time: {time_of_observation}")
    del series, time_of_observation, f, t, Sxx
    plt.clf()    
    plt.close()
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')
    gc.collect()

"""
Due to virtual environment and M1 macbook limitation i had to copy and modify lots of source code to create multitapered spectrogram.

Some of the source code is for creating "librosa" plots.

"""


# MULTITAPER SPECTROGRAM #
def multitaper_spectrogram(data, fs, frequency_range=None, time_bandwidth=5, num_tapers=None, window_params=None,
                           min_nfft=0, detrend_opt='linear', weighting='unity', plot_on=True, clim_scale = True, verbose=True, xyflip=False):
    
    # Set frequency range if not provided
    if frequency_range is None:
        frequency_range = [0, fs / 2]

     #  Process user input
    [data, fs, frequency_range, time_bandwidth, num_tapers,
     winsize_samples, winstep_samples, window_start,
     num_windows, nfft, detrend_opt, plot_on, verbose] = process_input(data, fs, frequency_range, time_bandwidth,
                                                                       num_tapers, window_params, min_nfft,
                                                                       detrend_opt, plot_on, verbose)
    # Set up spectrogram parameters
    [window_idxs, stimes, sfreqs, freq_inds] = process_spectrogram_params(fs, nfft, frequency_range, window_start,
                                                                          winsize_samples)

    # Split data into segments and preallocate
    data_segments = data[window_idxs]

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

    # set all but 1 arg of calc_mts_segment to constant (so we only have to supply one argument later)
    calc_mts_segment_plus_args = partial(calc_mts_segment, dpss_tapers=dpss_tapers, nfft=nfft, freq_inds=freq_inds,
                                         detrend_opt=detrend_opt, num_tapers=num_tapers, dpss_eigen=dpss_eigen,
                                         weighting=weighting, wt=wt)

    mt_spectrogram = np.apply_along_axis(calc_mts_segment_plus_args, 1, data_segments)

    # Compute one-sided PSD spectrum
    mt_spectrogram = np.asarray(mt_spectrogram)
    mt_spectrogram = mt_spectrogram.T
    dc_select = np.where(sfreqs == 0)
    nyquist_select = np.where(sfreqs == fs/2)
    select = np.setdiff1d(np.arange(0, len(sfreqs)), [dc_select, nyquist_select])

    mt_spectrogram = np.vstack([mt_spectrogram[dc_select[0], :], 2*mt_spectrogram[select, :],
                               mt_spectrogram[nyquist_select[0], :]]) / fs

    return mt_spectrogram, stimes, sfreqs

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

# LIBROSA SOURCE CODE (modified):
def mesh_coords(ax_type, coords, n, **kwargs):
    '''Compute axis coordinates'''

    if coords is not None:
        if len(coords) < n:
            raise ValueError('Coordinate shape mismatch: '
                                    '{}<{}'.format(len(coords), n))
        return coords

    coord_map = {'linear': __coord_fft_hz, 'hz': __coord_fft_hz, 'log': __coord_fft_hz, 'time': __coord_time}

    if ax_type not in coord_map:
        raise ValueError('Unknown axis type: {}'.format(ax_type))

    return coord_map[ax_type](n, **kwargs)

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

def decorate_axis(axis, ax_type):
    if ax_type == 'time':
        axis.set_major_formatter(TimeFormatter(lag=False))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Time')
    elif ax_type in ['linear', 'hz']:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_label_text('Hz')