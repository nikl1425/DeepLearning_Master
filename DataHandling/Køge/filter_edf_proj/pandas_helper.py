import pandas as pd
import mne
import numpy as np
import gc

from util import logging_info_txt

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

def insert_time_stamp(dataframe, file_start_time, frq, date_converter):
    print(f"file start meas date {file_start_time}")
    timestamp_ms = date_converter(file_start_time)
    period_row_increment_value =  (1 / int(frq)) * 1000
    dataframe.insert(0, "timestamp", [timestamp_ms + i * period_row_increment_value for i in dataframe.index])

class_mapping = {"Seizure": 1, "Preictal": 2, "Interictal": 3}

def get_class_map():
    return class_mapping


def insert_class_col(dataframe, sz_info_list, date_converter, save_filename, save_path, file_sample_rate, file_channel):
    print(f"sz_info_list: {sz_info_list}")
    
    for index, container in enumerate(sz_info_list):
        delay = container.delay * 1000
        duration = container.duration * 1000
        sz_start = date_converter(container.time_emu) + delay
        sz_end = sz_start + duration
        print(f"sz_start index = {sz_start}")
        print(f"sz_end: {sz_end}")
        preictal_start = sz_start - (15 * 60 * 1000)
        interictal_start = sz_start - (1 * 60 * 60 * 1000)
        interictal_end = sz_end + (1 * 60 * 60 * 1000)
        dataframe['timestamp'] = pd.to_numeric(dataframe['timestamp'])

        #INSERTING PREICTAL
        prei_df = dataframe[(dataframe['timestamp'] >= preictal_start) & (dataframe['timestamp'] < sz_start)]
        print(f"len prei: {len(prei_df)}")
        df_save_compress(f"Preictal_{index}_{save_filename}", save_path + "/Preictal", prei_df)
        logging_info_txt(f"Preictal_{index}_{save_filename}", save_path, file_sample_rate, file_channel)
        
        #INSERTING SEIZURE CLASS
        sz_df = dataframe[(dataframe['timestamp'] >= sz_start) & (dataframe['timestamp'] < sz_end)]
        print(f"len sz: {len(sz_df)}")
        df_save_compress(f"Seizure_{index}_{save_filename}", save_path + "/Seizure", sz_df)
        logging_info_txt(f"Seizure_{index}_{save_filename}", save_path, file_sample_rate, file_channel)

        #INSERTING INTERICTAL
        pre_int_df = dataframe[(dataframe['timestamp'] >= interictal_start) & (dataframe['timestamp'] < preictal_start)]
        print(f"len pre int: {len(pre_int_df)}")
        df_save_compress(f"PreInt_{index}_{save_filename}", save_path + "/Interictal", pre_int_df)
        logging_info_txt(f"PreInt_{index}_{save_filename}", save_path, file_sample_rate, file_channel)

        post_int_df = dataframe[(dataframe['timestamp'] >= sz_end) & (dataframe['timestamp'] < interictal_end)]
        print(f"len post int: {len(post_int_df)}")
        df_save_compress(f"PostInt_{index}_{save_filename}", save_path + "/Interictal", post_int_df)
        logging_info_txt(f"PostInt_{index}_{save_filename}", save_path, file_sample_rate, file_channel)

        #print(f"after = len df: {len(dataframe)} values class: \n {dataframe['class'].value_counts()}")
        
        # clean up
        del pre_int_df, post_int_df, sz_df, prei_df
        gc.collect()


def df_save_compress(filename, save_path, df):
    df.to_csv(f"{save_path}/{filename}.csv")

