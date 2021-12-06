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
from spectrogram import spec_transform_save_to_folder

# Globals:
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
files = []
for folder in os.listdir(external_hardisk_drive_path + "Køge_03/EEG/Csv/"):
    for filename in os.listdir(external_hardisk_drive_path + "Køge_03/EEG/Csv/" + folder + "/"):
        fullpath = external_hardisk_drive_path + "Køge_03/EEG/Csv/" + folder + "/" + filename
        files.append(fullpath)

window_path = external_hardisk_drive_path + "Køge_03/Windows/EEG/"
info_txt_file_path = external_hardisk_drive_path + "/Køge_03/EEG/info.txt"

def create_spec():
    count = 0
    info_obj = get_info_text(info_txt_file_path)
    print(f"info_obj: {info_obj}")
    for filename in files:
        print("started file: " + str(filename) + " index: " + str(count))
        global FREQ
        FREQ = [find_frq(x) for x in info_obj if find_filename(x) in filename][0] if len([find_frq(x) for x in info_obj if find_filename(x) in filename]) > 0 else 500
        ch = [find_channel(x) for x in info_obj if find_filename(x) in filename][0]
        print(f"FREQ Set: {FREQ}")
        df, selected_channels = read_compressed_df(filename, ch)
        if FREQ >= 1000:
            df = df.iloc[::2,:]

        patient = re.search('patient_(.*)_date_', filename).group(1)
        print(patient)
        for channel in selected_channels:
            print(f"channel: " + str(channel))
            if len(df) > 0 and df.empty == False:
                if "Interictal" in filename:
                    inter_win = [get_window(channel=channel,start_index=i, data=df) for i in range(get_max_window_iteration(df, 4))]
                    for index, window in enumerate(inter_win):
                        spec_transform_save_to_folder(channel=channel, index=index, win=window, patient=patient, save_path=window_path, patient_state = "interictal")
                        del window
                    del inter_win

                if "Seizure" in filename:
                    sz_win = [get_window(channel=channel, start_index=i, data=df, is_sezure=True) for i in range(get_max_window_iteration(df, 2))]
                    for index, window in enumerate(sz_win):
                        spec_transform_save_to_folder(channel=channel, index=index, win=window, patient=patient, save_path=window_path, patient_state="seizure")
                        del window
                    del sz_win

                if  "Preictal" in filename:
                    prei_one_win = [get_window(channel=channel,start_index=i, data=df) for i in range(get_max_window_iteration(df, 4))]
                    for index, window in enumerate(prei_one_win):
                        spec_transform_save_to_folder(channel=channel, index=index, win=window, patient=patient, save_path=window_path, patient_state="prei_one")
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
    create_spec()

   

    