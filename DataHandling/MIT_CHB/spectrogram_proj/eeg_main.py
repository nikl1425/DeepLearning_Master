import sys
import os
import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import re
from datetime import datetime
import matplotlib
import os
from matplotlib import pyplot as plt
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

# Import custom function:
from pandas_helper import get_max_window_iteration, get_window, read_compressed_df
from util import get_info_text, find_frq, find_filename, find_channel, logging_info_txt
from spectrogram import spec_save_to_folder, multitaper_spec_save_to_folder

# Globals:
cwd = os.getcwd()
print(cwd)
FREQ = 256
database_path = database_path = 'Dataset/CHB-MIT/'
filtered_database_path = '../Dataset/CHB-MIT/Filtered-chb-mit/'
compressed_file_type = ".csv"

database_path = 'Dataset/CHB-MIT/'
filtered_database_path = '../Dataset/CHB-MIT/Filtered-chb-mit/'
save_csv_path = "../DataSet/CHB-MIT/filtered_df_csv"
external_hardisk_drive_path = os.path.dirname('/Volumes/NHR HDD/CHB-MIT/')
csv_path = external_hardisk_drive_path + "/19-12-csv/"


files = []
for folder in os.listdir(csv_path):
    for filename in os.listdir(csv_path + folder + "/"):
        fullpath = csv_path + folder + "/" + filename
        files.append(fullpath)

window_path = external_hardisk_drive_path + "/Windows_19-12/Windows/Train/"


def create_spec():
    count = 0
    for filename in files:
        print("started file: " + str(filename) + " index: " + str(count))
        df, selected_channels = read_compressed_df(filename, ['FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1','FZ-CZ', 'CZ-PZ'])

        patient = "patient_" + str(re.search('/chb(.*)_', filename).group(1))
        print(patient)
        for channel in selected_channels:
            print(f"channel: " + str(channel))
            if len(df) > 0 and df.empty == False:
                if "Interictal" in filename:
                    inter_win = [get_window(channel=channel,start_index=i, data=df) for i in range(get_max_window_iteration(df, 4))]
                    for index, window in enumerate(inter_win):
                        spec_save_to_folder(channel=channel, index=index, win=window, patient=patient, save_path=window_path, patient_state = "interictal")
                        del window
                    del inter_win

                if "Seizure" in filename:
                    sz_win = [get_window(channel=channel, start_index=i, data=df, is_sezure=True) for i in range(get_max_window_iteration(df, 2))]
                    for index, window in enumerate(sz_win):
                        spec_save_to_folder(channel=channel, index=index, win=window, patient=patient, save_path=window_path, patient_state="seizure")
                        del window
                    del sz_win

                if  "Preictal" in filename:
                    prei_one_win = [get_window(channel=channel,start_index=i, data=df) for i in range(get_max_window_iteration(df, 4))]
                    for index, window in enumerate(prei_one_win):
                        spec_save_to_folder(channel=channel, index=index, win=window, patient=patient, save_path=window_path, patient_state="prei_one")
                        del window
                    del prei_one_win

                Log_file_path = window_path + "progress.txt"
                #LOGGING:
                logging_info_txt(f"filename: {filename} channel: {channel} Created len: {len(df)} count = {count} : files left = {len(files) - count}\n", Log_file_path)

        gc.collect()
        count += 1
        print(f"memory usage = {psutil.virtual_memory().percent} : available memory = {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total}")
        print(f"filename: {filename} = done : count = {count} : files left = {len(files) - count} : time of creation = {datetime.now()}")
        del df

if __name__ == "__main__":
    for i, f in enumerate(files):
        print(f"index: {i} file: {f}")


    df, selected_channels = read_compressed_df("/Volumes/NHR HDD/CHB-MIT/19-12-csv/Interictal/chb01_04_sz_1467.csv", ['FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1','FZ-CZ', 'CZ-PZ'])

    
    # #!/usr/bin/python
    # import os

    # # getting the filename from the user
    # file_path = input("Enter filename:- ")

    # # checking whether file exists or not
    # if os.path.exists(file_path):
    #     # removing the file using the os.remove() method
    #     os.remove(file_path)
    # else:
    #     # file not found message
    #     print("File not found in the directory")
    #create_spec()

   

    