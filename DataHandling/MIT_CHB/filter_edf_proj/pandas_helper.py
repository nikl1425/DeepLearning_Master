import pandas as pd
import mne
import numpy as np
import gc

from util import logging_info_txt
from filter import apply_filter

def read_excel_to_df(path, sheet_name=None, sheet_mode=False):
    if sheet_mode:
        try:
            return pd.read_excel(path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Read excel failed: {e}")
    return pd.read_excel(path)

def format_unique_list(dataframe, row_val1, row_val2, index_val, index_val2):
    con_list = []

    for i, r in dataframe.iterrows():
        _v1 = r[row_val1]
        _v2 = r[row_val2]
        container = {index_val: _v1, index_val2: _v2}

        if container not in con_list:
            con_list.append(container)

    return con_list


class container():
    def __init__(self, delay, time_emu, duration, id):
        self.delay = delay
        self.time_emu = time_emu
        self.duration = duration
        self.id = id

def create_seizure_list(list_object, df_object, index_val1, index_val2):

    sz_info = []

    for c in list_object:
        f = c[index_val1]
        p = c[index_val2]
        print(f"create sz info f: {f}")
        cont_storage = []
        sz_count = 0
        for i, r in df_object.iterrows():
            if f == r['fullPath']:
                con = container(delay=r['delay'], time_emu=r['time_emu'], duration=r['seizureDuration'], id=r['SeizureID'])
                cont_storage.append(con)
        sz_info.append([p, f, cont_storage])
    return sz_info

edf_file_type = ".edf"
capitilize_edf_file_type = ".EDF"
bdf_file_type = ".BDF"

def read_edf_file(file_name, print_reader_info = False):
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

def insert_time_stamp(dataframe, file_start_time, FREQ):
    period_row_increment_value =  (1 / FREQ) * 1000
    dataframe.insert(0, "timestamp", [file_start_time + i * period_row_increment_value for i in dataframe.index])

class_mapping = {"Seizure": 1, "Preictal": 2, "Interictal": 3}

def get_class_map():
    return class_mapping

class_mapping = {"Seizure": 1, "Preictal": 2, "Interictal": 3}

def insert_class_col(dataframe_copy, sz_info_list, save_name, external_hardisk_drive_path, file_sample_rate):
    print(f"modtaget string: {sz_info_list}")

    channel_to_save = ['FP1-F7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']

    dataframe = dataframe_copy[channel_to_save]

    for channel in dataframe.columns:
        dataframe[channel] = apply_filter(dataframe[channel], file_sample_rate, low=True)
        find_log_min_max_welch(channel, dataframe, save_name, f"{external_hardisk_drive_path}/welch_info2.txt", file_sample_rate)

    quarter_ensurance = 15 * 60 * 256
    
    if "class" not in dataframe.columns:
        dataframe.insert(0, "class", np.nan)

    print(dataframe.columns)

    if len(sz_info_list) == 0:
        int = dataframe[0:-quarter_ensurance]
        df_save_compress(external_hardisk_drive_path + "/19-12-csv/" + "Interictal/" + f"{save_name}_none_sz", int)
        print(f"NO SZ len int {len(int)}")

    else:
        for item in sz_info_list:
            
            sz_start = item["sz_start"] * 256
            sz_end = item["sz_end"] * 256
            print(f"sz_start index = {sz_start}")
            print(f"sz_end: {sz_end}")
            preictal_start = sz_start - (15 * 60 * 256) if (sz_start - (15 * 60 * 256)) >= 0 else 0
            print(f"prei start {preictal_start}")
            interictal_start = preictal_start - (60 * 60 * 256) if preictal_start > 0 and preictal_start - (60 * 60 * 256) >= 0 else 0
            #print(f"dur: {sz_end - sz_start}")

            dataframe.loc[(dataframe.index > sz_start) & (dataframe.index < sz_end), 'class'] = class_mapping['Seizure']
            dataframe.loc[(dataframe.index > preictal_start) & (dataframe.index < sz_start), 'class'] = class_mapping['Preictal']
            dataframe.loc[(dataframe.index > interictal_start) & (dataframe.index < preictal_start), 'class'] = class_mapping['Interictal']


            dataframe['class'][sz_start : sz_end] = class_mapping["Seizure"]
            dataframe['class'][preictal_start : sz_start] = class_mapping["Preictal"]
            dataframe['class'][interictal_start : preictal_start] = class_mapping["Interictal"]
            print(dataframe['class'].value_counts())

            prei = dataframe[(dataframe.index >= preictal_start) & (dataframe.index < sz_start) & (dataframe['class'] != class_mapping["Seizure"])].copy()
            prei = dataframe[preictal_start:sz_start]
            df_save_compress(external_hardisk_drive_path + "/19-12-csv/" + "Preictal/" + f"{save_name}_prei_{item['sz_start']}", prei)
            print(f"prei len {len(prei)}")

            sz = dataframe[(dataframe.index >= sz_start) & (dataframe.index < sz_end)].copy()
            df_save_compress(external_hardisk_drive_path + "/19-12-csv/" + "Seizure/" + f"{save_name}_sz_{item['sz_start']}", sz)
            print(f"sz len {len(sz)}")

            int = dataframe[(dataframe.index >= interictal_start) & (dataframe.index < preictal_start) & (dataframe['class'] != class_mapping["Seizure"]) & (dataframe['class'] != class_mapping["Preictal"])].copy()
            if len(int) > 1*60*256:
                df_save_compress(external_hardisk_drive_path + "/19-12-csv/" + "Interictal/" + f"{save_name}_int_{item['sz_start']}", int)
            print(f"int len: {len(int)}")


def df_save_compress(filename, save_path, df):
    df.to_csv(f"{save_path}/{filename}.csv")

