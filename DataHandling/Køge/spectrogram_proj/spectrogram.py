import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import psutil
import gc
import matplotlib.colors as colors
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

def spec_transform_save_to_folder(index, win, channel, patient_state, patient, save_path, plot_title = False, FREQ = 500):
    interval = int(FREQ/2)   # ... the interval size,
    overlap = int(interval * 0.95)  # ... and the overlap of interval
    series = win[0]
    time_of_observation = win[1]

    try:
        series = np.array(series).astype(np.float)
    except Exception as e:
        print(f"error: {e}")
        print(f"patient_state: {patient_state} channel: {channel} index: {index} window: {series}")
    
    if plot_title:
        plt.title(f"{channel} : is_seizure = {patient_state} : {time_of_observation}")

    filt_series = apply_filter(series, FREQ)

    f, t, Sxx = signal.spectrogram(np.array(filt_series), fs=FREQ, nperseg=interval, noverlap=overlap, window='hann')
    #normalize_color= colors.LogNorm(vmin=np.amin(Sxx), vmax=np.amax(Sxx))      
    Sxx = 10*np.log10(Sxx) 
    plt.pcolormesh(t, f, Sxx, cmap='jet')
    plt.specgram(filt_series, cmap='jet', Fs=FREQ, NFFT=interval, noverlap=overlap)
    plt.axis('off')
    
    time_of_observation = str(time_of_observation).replace(":", "-")
    Log_file_path = save_path + "Log.txt"

    #LOGGING:
    logging_info_txt(f"patient: {patient} channel: {channel} time: {time_of_observation} FREQ: {FREQ} \n", Log_file_path)

    if patient_state == "seizure":
        plt.savefig(f'{save_path}Seizure/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
    elif patient_state == "interictal":
        plt.savefig(f'{save_path}Interictal/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
    elif patient_state == "prei_one":
        plt.savefig(f'{save_path}Preictal/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')

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

    