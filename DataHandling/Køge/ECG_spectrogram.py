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
FREQ = 500
database_path = 'Dataset/CHB-MIT/chb-mit-scalp-eeg-database-1.0.0/'
filtered_database_path = 'Dataset/CHB-MIT/Filtered-chb-mit/'
compressed_file_type = ".csv"


#internal
# files = ["../Dataset/CHB-MIT/filtered_df_csv/" + f for f in os.listdir( "../Dataset/CHB-MIT/filtered_df_csv/") if f.endswith(compressed_file_type)]
# internal_window_path = "/Users/niklashjort/Desktop/Notes/Speciale/projects/MIT_Model/test_images/"

#external
os.chdir("/volumes/")
external_hardisk_drive_path = '/Volumes/NHR HDD/'
files = [external_hardisk_drive_path + "Køge_02/ECG/" + f for f in os.listdir(external_hardisk_drive_path + "Køge_02/ECG/") if f.endswith(compressed_file_type)]
window_path = external_hardisk_drive_path + "Køge_02/Windows/EKG/"
Log_file_path = external_hardisk_drive_path + "/Køge_02/Windows/EKG/Log.txt"
info_txt_file_path = external_hardisk_drive_path + "/Køge_02/ECG/info.txt"


for f in files[0:1]:
    print(f)


#df = pd.read_csv(files[0])

#df['class'].value_counts()

#df.columns

def read_compressed_df(path):
    df = pd.read_csv(path, usecols=[1,2,3])
    df['class'] = df['class'].astype('int32')

    sz_df = df.loc[df['class'] == 1]

    prei_df = df.loc[df['class'] == 2]

    inter_df = df.loc[df['class'] == 3]
    
    channels = [item for item in list(sz_df.columns) if item != "class" if item != "timestamp"]

    return (sz_df, prei_df, inter_df, channels)

def get_window(channel, start_index, data, size = 4, overlap = 0, is_sezure = False, frequency = FREQ):
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

    print(f"Window: {start_index}")
    
    return [data[channel][start:end].tolist(), date_timestamp]

def get_max_window_iteration(dataframe, buffer):
    len_of_df = int(len(dataframe) // (buffer*FREQ))
    return len_of_df

def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.lfilter(b, a, data)
    return y


def logging_info_txt(string_to_write):
    file_object = open(Log_file_path, "a")
    file_object.write(string_to_write)
    file_object.close()

plt.figure(figsize=(10,10))       # Define the sampling frequency,

def spec_transform_save_to_folder(index, win, channel, patient_state, patient, plot_title = False):
    interval = int(FREQ/2)   # ... the interval size,
    overlap = int(interval * 0.95)  # ... and the overlap intervals

    series = win[0]
    time_of_observation = win[1]
    try:
        series = np.array(series).astype(np.float)
    except Exception as e:
        print(f"error: {e}")
        print(f"patient_state: {patient_state} channel: {channel} index: {index} window: {series}")
    if plot_title:
        plt.title(f"{channel} : is_seizure = {patient_state} : {time_of_observation}")

    series = butter_highpass_filter(series, 1, FREQ, order=6)
    
    
    try:             
        # plt.pcolormesh(t, f, Sxx, norm=normalize_color,
        #         cmap='jet')
        plt.specgram(series, cmap='jet', Fs=FREQ, NFFT=interval, noverlap=overlap)
        plt.axis('off')
        
        time_of_observation = str(time_of_observation).replace(":", "-")
        print(f"patient: {patient} time: {time_of_observation}")

        

        if patient_state == "seizure":
            plt.savefig(f'{window_path}Seizure/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
        elif patient_state == "interictal":
            plt.savefig(f'{window_path}Interictal/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
        elif patient_state == "prei_one":
            plt.savefig(f'{window_path}Preictal/{patient}_{index}_{channel}_{time_of_observation}.png', bbox_inches='tight')
    except:
        pass
    
    del series, time_of_observation
    plt.clf()    
    plt.close()
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')
    gc.collect()

def find_frq(str_obj):
    frq_found = re.search(r"<br> freq: (.*?) <br>", str_obj)
    if(frq_found):
        return int(float(frq_found.group(1)))

def find_filename(str_obj):
    frq_found = re.search(r"filename: (.*?) <br>", str_obj)
    if(frq_found):
        return (frq_found.group(1))

def get_info_text(path):
    with open(path, 'r') as f:
        str_container = ""
        formatted_str = []
        for line in f:
            str_container += str(line).replace("\n", "<br>")
            formatted_str = re.findall('(.*?)<br><br>', str_container)
        return formatted_str

def create_spec():
    count = 0
    info_obj = get_info_text(info_txt_file_path)
    for filename in files[9:10]:
        #logging_info_txt(f"len files {len(files)}")
        print("started file: " + str(filename) + " index: " + str(count))
        global FREQ
        FREQ = [find_frq(x) for x in info_obj if find_filename(x) in filename][0] if len([find_frq(x) for x in info_obj if find_filename(x) in filename]) > 0 else 500
        print(f"FREQ Set: {FREQ}")
        sz, prei_one, inter, selected_channels = read_compressed_df(filename)
        patient = re.search('patient_(.*)_date_', filename).group(1)
        print(patient)
        
        for channel in selected_channels:
            
            #LOGGING:
            logging_info_txt(f"patient: {patient} channel: {channel} FREQ: {FREQ} Value Counts: sz = {len(sz)} prei = {len(prei_one)} int = {len(inter)}")

            print(f"channel: " + str(channel))
            if len(inter) > 0 and inter.empty == False:
                print("inside")
                inter_win = [get_window(channel=channel,start_index=i, data=inter) for i in range(get_max_window_iteration(inter, 4))]
                for index, window in enumerate(inter_win):
                    spec_transform_save_to_folder(win=window, index=index, channel=channel, patient_state = "interictal", patient=patient)
                    del window
                del inter_win

            if len(sz) > 0 and sz.empty == False:
                sz_win = [get_window(channel=channel, start_index=i, data=sz, is_sezure=True) for i in range(get_max_window_iteration(sz, 2))]
                for index, window in enumerate(sz_win):
                    spec_transform_save_to_folder(channel=channel, index=index, win=window, patient_state="seizure", patient=patient)
                    del window
                del sz_win

            if len(prei_one) > 0 and prei_one.empty == False:
                prei_one_win = [get_window(channel=channel,start_index=i, data=prei_one) for i in range(get_max_window_iteration(prei_one, 4))]
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
    for i, f in enumerate(files):
        print(f"i: {i} f {f}")

    create_spec()
   

        

    