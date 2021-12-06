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

print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")

# Custom Modules
from pandas_helper import read_excel_to_df, format_unique_list, read_edf_file, insert_time_stamp, insert_class_col, df_save_compress, get_class_map, create_seizure_list
from util import convert_date_to_ms, downcast_dtypes, mem_usage, logging_info_txt



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


info_df = read_excel_to_df(info_df_path, sheet_name="NHR_EEG", sheet_mode=True)

for i, r in info_df.iterrows():
    patient_id = r['patientID']
    patient_folder = "Patient " + str(patient_id)
    EEG_file = r['fileName']
    for folder in os.listdir(database_path + "EEG"):
        if patient_folder == folder and EEG_file != 0:
            full_path_patient_file = database_path + f"EEG/{patient_folder}/{EEG_file}"
            info_df.loc[i, "fullPath"] = full_path_patient_file

info_list = format_unique_list(info_df, "patientID", "fullPath", "ID", "File")

file_sz_info = create_seizure_list(info_list, info_df, "File", "ID")

class_mapping = get_class_map()

def run_save_pd_csv():
    for e in file_sz_info:
        print(f"patient_id: {e[0]}")
        print(f"file_name: {e[1]}")

        file_path = e[1]
        df, data_info = read_edf_file(file_path)
        df = downcast_dtypes(df)
        file_sample_rate = data_info["sfreq"]
        file_meas_date = data_info["meas_date"]
        file_channel = data_info['ch_names']
        relevant_channels = file_channel[0:2]
        print(f"freq: {file_sample_rate} meas: {file_meas_date} channels: {relevant_channels}")
        
        insert_time_stamp(df, file_meas_date, file_sample_rate, convert_date_to_ms)

        save_format_date = str(file_meas_date).replace(":", "").replace("+", "").replace("/","")
        save_file_name = f"patient_{e[0]}_date_{save_format_date}"

        insert_class_col(df, e[2], convert_date_to_ms, save_file_name, save_csv_path, file_sample_rate, relevant_channels)

        #LOGGING:
        #logging_info_txt(save_file_name, save_csv_path, file_sample_rate, file_channel)

        #Only keep rows containing class:
        #df = df[df['class'].isin([class_mapping['Interictal'], class_mapping['Seizure'], class_mapping['Preictal']])]

        #SAVE TO CSV
        #df_save_compress(save_file_name, save_csv_path, df)

        #Memory:
        del df, data_info
        gc.collect()

        print("DONE")

if __name__ == "__main__":
    run_save_pd_csv()

