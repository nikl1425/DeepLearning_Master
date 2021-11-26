
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
import glob
matplotlib.use('Qt5Agg')
#matplotlib.use('agg')

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

edf_file_type = ".edf"
capitilize_edf_file_type = ".EDF"
bdf_file_type = ".BDF"
patient_one_path = 'chb04/'

# INTERNAL PATH:
database_path = "/Users/niklashjort/Desktop/Notes/Speciale/projects/Dataset/EMU_monitor(ruc)/"
info_df_path = "/Users/niklashjort/Desktop/Notes/Speciale/projects/Dataset/EMU_monitor(ruc)/NHR_Eventlist_RUC.xlsx"
save_csv_path = "/Users/niklashjort/Desktop/Notes/Speciale/projects/Dataset/EMU_monitor(ruc)/NHR/EEG/"


# EXTERNAL PATH:
# os.chdir("/Volumes/NHR HDD/")
# database_path = "EMU_monitor(ruc)/"
# info_df_path = database_path + "NHR_Eventlist_RUC.xlsx"
# save_csv_path = "Køge/EEG_csv_filtered/"


info_df = pd.read_excel(info_df_path, sheet_name="NHR_EEG")


info_df.head()


for i, r in info_df.iterrows():
    patient_id = r['patientID']
    patient_folder = "Patient " + str(patient_id)
    EEG_file = r['fileName']
    for folder in os.listdir(database_path + "EEG"):
        if patient_folder == folder and EEG_file != 0:
            full_path_patient_file = database_path + f"EEG/{patient_folder}/{EEG_file}"
            info_df.loc[i, "fullPath"] = full_path_patient_file


info_list = []

for i, r in info_df.iterrows():
    patient_id = r['patientID']
    patient_file = r['fullPath']
    container = {"ID": patient_id, "File": patient_file}

    if container not in info_list:
        info_list.append(container)


class container():
    def __init__(self, delay, time_emu, duration, id):
        self.delay = delay
        self.time_emu = time_emu
        self.duration = duration
        self.id = id

file_sz_info = []

for c in info_list:
    f = c['File']
    p = c['ID']
    print(f"info_list f: {f}")
    cont_storage = []
    sz_count = 0
    for i, r in info_df.iterrows():
        if f == r['fullPath']:
            con = container(delay=r['delay'], time_emu=r['time_emu'], duration=r['seizureDuration'], id=r['SeizureID'])
            cont_storage.append(con)
    file_sz_info.append([p, f, cont_storage])

def ReadEdfFile(file_name, print_reader_info = False):
    if edf_file_type in file_name or capitilize_edf_file_type in file_name:
        if(print_reader_info):
            data = mne.io.read_raw_edf(file_name)
            raw_data = data.get_data()
            converted_raw = pd.DataFrame(raw_data.transpose(), columns=data.ch_names)
            print(data.info)
            return converted_raw, data.info
        else:
            data = mne.io.read_raw_edf(file_name, verbose='error')
            raw_data = data.get_data()
            converted_raw = pd.DataFrame(raw_data.transpose(), columns=data.ch_names)
            return converted_raw, data.info
    if bdf_file_type in file_name:
        if(print_reader_info):
            data = mne.io.read_raw_bdf(file_name)
            raw_data = data.get_data()
            converted_raw = pd.DataFrame(raw_data.transpose(), columns=data.ch_names)
            print(data.info)
            return converted_raw, data.info
        else:
            data = mne.io.read_raw_bdf(file_name, verbose='error')
            raw_data = data.get_data()
            converted_raw = pd.DataFrame(raw_data.transpose(), columns=data.ch_names)
            return converted_raw, data.info


def convert_date_to_ms(date_time):
    date_time = str(date_time)
    if "+" in str(date_time):
        date_time = str(date_time).split("+")[0]

    try:
        timestamp_ms = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S').timestamp() * 1000
    except:
        timestamp_ms = datetime.strptime(date_time, '%d-%m-%Y %H:%M:%S').timestamp() * 1000
    return timestamp_ms

def insert_time_stamp(dataframe, file_start_time, frq):
    timestamp_ms = convert_date_to_ms(file_start_time)
    period_row_increment_value =  (1 / int(frq)) * 1000
    dataframe.insert(0, "timestamp", [timestamp_ms + i * period_row_increment_value for i in dataframe.index])


class_mapping = {"Seizure": 1, "Preictal": 2, "Interictal": 3}

def insert_class_col(dataframe, sz_info_list, class_mapping):
    print(f"sz_info_list: {sz_info_list}")
    
    if "class" not in dataframe.columns:
        dataframe.insert(0, "class", np.nan)

    if len(sz_info_list) == 0:
        dataframe.loc[(dataframe['class'] != "Seizure") & (dataframe['class'] != "Preictal I") & (dataframe['class'] != "Preictal II"), "class"] = "Interictal"
    else:
        for container in sz_info_list:
            delay = container.delay * 1000
            duration = container.duration * 1000
            sz_start = convert_date_to_ms(container.time_emu) + delay
            sz_end = sz_start + duration
            print(f"sz_start index = {sz_start}")
            print(f"sz_end: {sz_end}")
            preictal_start = sz_start - (15 * 60 * 1000)
            interictal_start = sz_start - (2 * 60 * 60 * 1000)
            interictal_end = sz_end + (2 * 60 * 60 * 1000)
            dataframe['timestamp'] = pd.to_numeric(dataframe['timestamp'])

            #INSERTING PREICTAL
            dataframe.loc[(dataframe['class'] != class_mapping['Seizure']) & (dataframe['timestamp'] >= preictal_start) & (dataframe['timestamp'] < sz_start), "class"] = class_mapping['Preictal']

            #INSERTING SEIZURE CLASS
            dataframe.loc[(dataframe['timestamp'] >= sz_start) & (dataframe['timestamp'] < sz_end), "class"] = class_mapping['Seizure']

            #INSERTING INTERICTAL
            dataframe.loc[(dataframe['class'] != class_mapping['Seizure']) & (dataframe['class'] != class_mapping['Preictal']) & (dataframe['timestamp'] >= interictal_start) & (dataframe['timestamp'] < interictal_end), "class"] = class_mapping['Interictal']

            print(f"after = len df: {len(dataframe)} values class: \n {dataframe['class'].value_counts()}")
    gc.collect()


def logging_info_txt(csv_file_name, save_path, freq, channels):
    file_object = open(save_path + "info.txt", "a")
    file_object.write(f"\nfilename: {csv_file_name} \n freq: {freq} \n channels: {channels} \n")
    file_object.close()

def df_save_compress(filename, df):
    df.to_csv(f"{save_csv_path}/{filename}.csv")

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def downcast_dtypes(df):
    _start = df.memory_usage(deep=True).sum() / 1024 ** 2
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    _end = df.memory_usage(deep=True).sum() / 1024 ** 2
    saved_time = (_start - _end) / _start * 100
    #print(f"Saved: {saved_time:.2f}%")
    return df

def run_save_pd_csv():
    for e in file_sz_info:
        print(f"patient_id: {e[0]}")
        print(f"file_name: {e[1]}")

        file_path = e[1]
        df, data_info = ReadEdfFile(file_path)
        df = downcast_dtypes(df)
        file_sample_rate = data_info["sfreq"]
        file_meas_date = data_info["meas_date"]
        file_channel = data_info['ch_names']
        relevant_channels = file_channel[0:2]
        print(f"freq: {file_sample_rate} meas: {file_meas_date} channels: {relevant_channels}")
        
        insert_time_stamp(df, file_meas_date, file_sample_rate)
        insert_class_col(df, e[2])

        save_format_date = str(file_meas_date).replace(":", "").replace("+", "").replace("/","")
        save_file_name = f"patient_{e[0]}_date_{save_format_date}"

        #LOGGING:
        logging_info_txt(save_file_name, file_sample_rate, file_channel)

        #Only keep rows containing class:
        df = df[df['class'].isin([class_mapping['Interictal'], class_mapping['Seizure'], class_mapping['Preictal']])]

        #SAVE TO CSV
        df_save_compress(save_file_name, df)

        #Memory:
        del df, data_info
        gc.collect()

        print("DONE")
    


if __name__ == "__main__":
    print(f"class seizure mapping: {class_mapping['Seizure']} type: {type(class_mapping['Seizure'])}")

    run_save_pd_csv()