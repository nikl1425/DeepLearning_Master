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
import random
import gc
import os
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Sequential
from matplotlib import pyplot as plt

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")


from util import *
from pandas_helper import *

cwd = os.getcwd()
FREQ = 256
database_path = 'Dataset/CHB-MIT/chb-mit-scalp-eeg-database-1.0.0/'
save_csv_path = "../DataSet/CHB-MIT/filtered_df_csv"
external_hardisk_drive_path = os.path.dirname('/Volumes/NHR HDD/CHB-MIT/')
filtered_database_path = external_hardisk_drive_path + "/Filtered-chb-mit/"
welch_treshold_info_path = "/welch_info.txt"





def run_save_pd_csv():
    pat_folders = get_all_patient_folder_names(filtered_database_path)

    for patient in pat_folders:
        #print(patient)
        current_patient = patient
        info_txt_path, edf_files = get_all_file_names(current_patient)
        # read & extract information
        info_txt = read_format_info_file(info_txt_path)
        sort_info_txt = sort_remove_files(info_txt)
        for line in sort_info_txt:
            print(f"l: {line}")
            edf_info_container = FileInformationContainer(line)
        # print(f"EDF_CONTAINER: info_string passed {edf_info_container.information_str}")
        # print(f"EDF_CONTAINER: ts_start {edf_info_container.time_start}")
        # print(f"EDF_CONTAINER: sz_info {edf_info_container.sz_info}")
            selected_edf_path = [x for x in edf_files if (edf_info_container.file_name in x)][0]
            edf_df, info = read_edf_file(selected_edf_path)
            if edf_df is not None:
                #edf_df = downcast_dtypes(edf_df)
                #insert_time_stamp(edf_df, edf_info_container.time_start)
                #print(f"info list = {edf_info_container.sz_info}")
                insert_class_col(edf_df, edf_info_container.sz_info, edf_info_container.file_name, FREQ)
                #print(f"filename: {edf_info_container.file_name} classes: {edf_df['class'].value_counts()}")
                print("-------------------------------------------------------")
                #print(edf_info_container.file_name)
                #df_save_compress(edf_info_container.file_name, edf_df)
                #print(f"saved: {edf_info_container.file_name}")

if __name__ == "__main__":

    run_save_pd_csv()

