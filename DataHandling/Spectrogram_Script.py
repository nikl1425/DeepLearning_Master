# %%
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
pd.io.parquet.get_engine('auto').__class__
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
filted_db_parquet_path = "Dataset/CHB-MIT/dataframe-parquet"
win_chb01_path = "Dataset/win_chb_01/"
external_hardisk_drive_path = '/Volumes/LaCie/Database/'
edf_file_type = ".edf"
compressed_file_type = ".csv"

#internal
files = ["../Dataset/CHB-MIT/filtered_df_csv/" + f for f in os.listdir( "../Dataset/CHB-MIT/filtered_df_csv/") if f.endswith(compressed_file_type)]

#external
# os.chdir("/volumes/")
# files = [external_hardisk_drive_path + "filtered_df_csv/" + f for f in os.listdir(external_hardisk_drive_path + "filtered_df_csv/") if f.endswith(compressed_file_type)]

test_patient = "chb01"

files = [f for f in files if test_patient in f]


for f in files[0:1]:
    print(f)

def remove_cols(dataframe, col_start = 0, col_end = 0):
    if col_end == 0:
        col_end = len(dataframe.columns) - 1
    
    dataframe = dataframe.iloc[: , col_start: col_end]
    return dataframe


df = pd.read_csv(files[0])


matches = ['.-0','.-1', '.-2', '.-3', '.-4', 'STI 014', 'Unnamed: 0']

def read_compressed_df(path, channel="F3-C3"):
    df = pd.read_csv(path)
    
  
    if any(x in df.columns for x in matches):
        for col_name in matches:
            try:
                df.drop(columns=col_name, inplace=True)
                
            except:
                pass

    #df = df[['timestamp', 'class', channel]]

    sz_df = df.loc[df['class'] == "seizure"].reset_index(drop=True)
    sz_df = remove_cols(sz_df)

    prei_one_df = df.loc[df['class'] == "Preictal I"]
    prei_one_df = remove_cols(prei_one_df)

    prei_two_df = df.loc[df['class'] == "Preictal II"]
    prei_two_df = remove_cols(prei_two_df)

    inter_df = df.loc[df['class'] == "Interictal"]
    inter_df = remove_cols(inter_df)

    channels = [item for item in list(sz_df.columns) if item != "class" if item != "timestamp"]

    return (sz_df, prei_one_df, prei_two_df, inter_df, channels)


def get_window(channel, start_index, data, size = 4, overlap = 0, is_sezure = False, frequency = 256):
    if(is_sezure):
        overlap = 2
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
interval = Fs        # ... the interval size,
overlap = Fs * 0.95  # ... and the overlap intervals

def spec_transform_save_to_folder(index, win, channel, patient_state, patient, plot_title = False):
    series = win[0]
    time_of_observation = win[1]
    try:
        series = np.array(series).astype(np.float)
    except Exception as e:
        print(f"error: {e}")
        print(f"patient_state: {patient_state} channel: {channel} index: {index} window: {series}")
    denoised_series = denoise_wavelet(series, method='BayesShrink',wavelet='db6', mode='hard',rescale_sigma=True, multichannel=False, wavelet_levels=3)
    if plot_title:
        plt.title(f"{channel} : is_seizure = {patient_state} : {time_of_observation}")


    f, t, Sxx = signal.spectrogram(denoised_series, fs=Fs, nperseg=interval, noverlap=overlap)
                         
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), cmap='jet', shading='auto')
    plt.axis('off')

    #freqs, psd = signal.periodogram(denoised_series, 256, scaling='density')

    #s,f,t,im = plt.specgram(denoised_series,Fs=FREQ,cmap='jet', NFFT=int(FREQ/4), noverlap=int(FREQ/8))
    
    time_of_observation = str(time_of_observation).replace(":", "-")
    print(time_of_observation)

    if patient_state == "seizure":
        plt.savefig(f'/Volumes/LaCie/Database/All_channel_chb_01/Train/Seizure/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
    elif patient_state == "interictal":
        plt.savefig(f'/Volumes/LaCie/Database/All_channel_chb_01/Train/Interictal/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
    elif patient_state == "prei_one":
        plt.savefig(f'/Volumes/LaCie/Database/All_channel_chb_01/Train/Preictal_One/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
    elif patient_state == "prei_two":
        plt.savefig(f'/Volumes/LaCie/Database/All_channel_chb_01/Train/Preictal_Two/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
    
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

count = 0

def create_save_spectrogram():
    for filename in files:
        print("started file: " + str(filename) + " index: " + str(count))
        sz, prei_one, prei_two, inter, selected_channels = read_compressed_df(filename)
        patient = re.search('/Volumes/LaCie/Database/filtered_df_csv/(.*).csv', filename).group(1)
        print(patient)
        for channel in selected_channels:
            print(f"channel: " + str(channel))
            if len(inter) > 0 and inter.empty == False:
                inter_win = [get_window(channel=channel,start_index=i, data=inter) for i in range(get_max_window_iteration(inter, 4))]
                for index, window in enumerate(inter_win):
                    spec_transform_save_to_folder(win=window, index=index, channel=channel, patient_state = "interictal", patient=patient)
                    del window
                    gc.collect()
                del inter_win

            if len(sz) > 0 and sz.empty == False:
                sz_win = [get_window(channel=channel, start_index=i, data=sz, is_sezure=True) for i in range(get_max_window_iteration(sz, 2))]
                for index, window in enumerate(sz_win):
                    spec_transform_save_to_folder(channel=channel, index=index, win=window, patient_state="seizure", patient=patient)
                    del window
                    gc.collect()
                del sz_win

            if len(prei_one) > 0 and prei_one.empty == False:
                prei_one_win = [get_window(channel=channel,start_index=i, data=prei_one) for i in range(get_max_window_iteration(prei_one, 4))]
                for index, window in enumerate(prei_one_win):
                    spec_transform_save_to_folder(channel=channel, index=index, win=window, patient_state="prei_one", patient=patient)
                    del window
                    gc.collect()
                del prei_one_win

            if len(prei_two) > 0 and prei_two.empty == False:
                prei_two_win = [get_window(channel=channel, start_index=i, data=prei_two) for i in range(get_max_window_iteration(prei_two, 4))]
                for index, window in enumerate(prei_two_win):
                    spec_transform_save_to_folder(channel=channel, index=index, win=window, patient_state="prei_two", patient=patient)
                    del window
                    gc.collect()
                del prei_two_win
        gc.collect()
        count += 1
        print(f"memory usage = {psutil.virtual_memory().percent} : available memory = {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total}")
        print(f"filename: {filename} = done : count = {count} : files left = {len(files) - count} : time of creation = {datetime.now()}")
        del sz, prei_one, prei_two, inter

if __name__ == "__main__":
    filename = files[1]
    print(filename)
    sz, prei_one, prei_two, inter, selected_channels = read_compressed_df(filename)
    for i, x in enumerate(selected_channels):
        print(f"{x} : {i}")