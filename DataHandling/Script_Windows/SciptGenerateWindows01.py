
# What version of Python do you have?
import sys
import os
import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import numpy as np
import re
import mne
import pathlib
import openpyxl
from datetime import datetime
import pytz
import matplotlib
import random
import os
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Sequential
from matplotlib import pyplot as plt
from scipy import signal
plt.ioff()
import psutil
import gc
matplotlib.use('agg')

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

cwd = os.getcwd()
print(cwd)
FREQ = 256
database_path = 'Dataset/CHB-MIT/chb-mit-scalp-eeg-database-1.0.0/'
filtered_database_path = 'Dataset/CHB-MIT/Filtered-chb-mit/'
external_hardisk_drive_path = '/Volumes/NHR HDD/'
window_path = "Windows_03/"
edf_file_type = ".edf"
compressed_file_type = ".csv"

# #internal
# files = ["../Dataset/CHB-MIT/filtered_df_csv/" + f for f in os.listdir( "../Dataset/CHB-MIT/filtered_df_csv/") if f.endswith(compressed_file_type)]
# internal_window_path = "/Users/niklashjort/Desktop/Notes/Speciale/projects/MIT_Model/test_images/"

#external
os.chdir(external_hardisk_drive_path)
files = [external_hardisk_drive_path + "filtered_df_csv/" + f for f in os.listdir(external_hardisk_drive_path + "filtered_df_csv/") if f.endswith(compressed_file_type)]

test_patient = "chb02"

files = [f for f in files]

def remove_cols(dataframe, col_start = 0, col_end = 0):
    if col_end == 0:
        col_end = len(dataframe.columns) - 1
    
    dataframe = dataframe.iloc[: , col_start: col_end]
    return dataframe


df = pd.read_csv(files[0])


df['class'].value_counts()


df.columns


matches = ['.-0','.-1', '.-2', '.-3', '.-4', 'STI 014', 'Unnamed: 0']
selected_channels = ['F3-C3', 'T8-P8', 'FP2-F4']

def read_compressed_df(path, channel="F3-C3"):
    df = pd.read_csv(path)
    
  
    if any(x in df.columns for x in matches):
        for col_name in matches:
            try:
                df.drop(columns=col_name, inplace=True)
                
            except:
                pass

    sz_df = df.loc[df['class'] == "seizure"].reset_index(drop=True)
    sz_df = remove_cols(sz_df)

    prei_one_df = df.loc[df['class'] == "Preictal"]
    prei_one_df = remove_cols(prei_one_df)

    inter_df = df.loc[df['class'] == "Interictal"]
    inter_df = remove_cols(inter_df)

    channels = [item for item in list(sz_df.columns) if item != "class" if item != "timestamp" if item in selected_channels]

    return (sz_df, prei_one_df, inter_df, channels)

def get_window(channel, start_index, data, size = 2, overlap = 0, is_sezure = False, frequency = 256):
    if(is_sezure):
        overlap = 1
    else:
        overlap = overlap
    
    start = start_index * (size - overlap) * frequency
    end = start + (size * frequency)

    date_timestamp = ""

    try:
        date_timestamp = datetime.fromtimestamp((data['timestamp'][start:start+1]/1000).tolist()[0]).strftime('%H:%M:%S')
    except:
        date_timestamp = "datetime cannot be converted"
    
    return [data[channel][start:end].tolist(), date_timestamp]

def get_max_window_iteration(dataframe, buffer):
    len_of_df = int(len(dataframe) / (buffer*256))
    return len_of_df

plt.figure(figsize=(10,10))
Fs = 256         # Define the sampling frequency,
interval = 256   # ... the interval size,
overlap = interval * 0.9  # ... and the overlap intervals
filter_order = 7
frequency_cutoff = 60
sampling_frequency = 256

import matplotlib.colors as colors

def spec_transform_save_to_folder(index, win, channel, patient_state, patient, plot_title = False):
    series = win[0]
    time_of_observation = win[1]
    try:
        series = np.array(series).astype(np.float)
    except Exception as e:
        print(f"error: {e}")
        print(f"patient_state: {patient_state} channel: {channel} index: {index} window: {series}")
    denoised_series = denoise_wavelet(series, method='BayesShrink',wavelet='db8', mode='hard',rescale_sigma=True, multichannel=False, wavelet_levels=3)
    if plot_title:
        plt.title(f"{channel} : is_seizure = {patient_state} : {time_of_observation}")
    
    # b, a = signal.butter(filter_order, frequency_cutoff, btype='low', output='ba', fs=sampling_frequency)
    # low_pass_signal = signal.filtfilt(b, a, denoised_series)

    f, t, Sxx = signal.spectrogram(denoised_series, fs=Fs, nperseg=interval, noverlap=overlap)

    #Sxx = 10*np.log10(Sxx)
    normalize_color= colors.LogNorm(vmin=np.amin(Sxx), vmax=np.amax(Sxx))
    try:             
        # plt.pcolormesh(t, f, Sxx, norm=normalize_color,
        #         cmap='jet')
        #plt.specgram(low_pass_signal, cmap='jet', Fs=256)
        plt.pcolormesh(t, f, np.abs(Sxx), norm=normalize_color, cmap='jet')# Plot the result
        #plt.ylim(0,70)
        plt.axis('off')

        #freqs, psd = signal.periodogram(denoised_series, 256, scaling='density')

        #s,f,t,im = plt.specgram(denoised_series,Fs=FREQ,cmap='jet', NFFT=int(FREQ/4), noverlap=int(FREQ/8))
        
        time_of_observation = str(time_of_observation).replace(":", "-")
        print(time_of_observation)

        if patient_state == "seizure":
            plt.savefig(f'{external_hardisk_drive_path + window_path}Seizure/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
        elif patient_state == "interictal":
            plt.savefig(f'{external_hardisk_drive_path + window_path}Interictal/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
        elif patient_state == "prei_one":
            plt.savefig(f'{external_hardisk_drive_path + window_path}Preictal/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
    except:
        pass
    
    del series, denoised_series, time_of_observation, f, t, Sxx
    plt.clf()    
    plt.close()
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')
    gc.collect()



def create_spec():
    count = 0
    for filename in files:
        print("started file: " + str(filename) + " index: " + str(count))
        sz, prei_one, inter, selected_channels = read_compressed_df(filename)
        patient = re.search('/Volumes/NHR HDD/filtered_df_csv/(.*).csv', filename).group(1)
        print(patient)
        for channel in selected_channels:
            print(f"channel: " + str(channel))
            if len(inter) > 0 and inter.empty == False:
                inter_win = [get_window(channel=channel,start_index=i, data=inter) for i in range(get_max_window_iteration(inter, 2))]
                for index, window in enumerate(inter_win):
                    spec_transform_save_to_folder(win=window, index=index, channel=channel, patient_state = "interictal", patient=patient)
                    del window
                del inter_win

            if len(sz) > 0 and sz.empty == False:
                sz_win = [get_window(channel=channel, start_index=i, data=sz, is_sezure=True) for i in range(get_max_window_iteration(sz, 1))]
                for index, window in enumerate(sz_win):
                    spec_transform_save_to_folder(channel=channel, index=index, win=window, patient_state="seizure", patient=patient)
                    del window
                del sz_win

            if len(prei_one) > 0 and prei_one.empty == False:
                prei_one_win = [get_window(channel=channel,start_index=i, data=prei_one) for i in range(get_max_window_iteration(prei_one, 2))]
                for index, window in enumerate(prei_one_win):
                    spec_transform_save_to_folder(channel=channel, index=index, win=window, patient_state="prei_one", patient=patient)
                    del window
                del prei_one_win

        gc.collect()
        count += 1
        print(f"memory usage = {psutil.virtual_memory().percent} : available memory = {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total}")
        print(f"filename: {filename} = done : count = {count} : files left = {len(files) - count} : time of creation = {datetime.now()}")
        del sz, prei_one , inter

if __name__ == "__main__":
    create_spec()