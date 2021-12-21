from datetime import datetime
import pandas as pd
import numpy as np
from scipy.signal import welch
import gc

def convert_date_to_ms(date_time):
    date_time = str(date_time)
    if "+" in str(date_time):
        date_time = str(date_time).split("+")[0]

    try:
        timestamp_ms = datetime.utcnow().strptime(date_time, '%Y-%m-%d %H:%M:%S').timestamp() * 1000
    except:
        timestamp_ms = datetime.utcnow().strptime(date_time, '%d-%m-%Y %H:%M:%S').timestamp() * 1000
    return timestamp_ms

def logging_info_txt(csv_file_name, save_path, freq, channels):
    print("logging")
    file_object = open(save_path + "info.txt", "a")
    file_object.write(f"\nfilename: {csv_file_name} \n freq: {freq} \n channels: {channels} \n")
    file_object.close()

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

def find_log_min_max_welch(channels, signal,  filename, log_file, freq=500):
    for c in channels:
        _, Pxx = welch(signal[c], fs=freq, window='hanning', detrend=False)
        Pxx_den_db = 10*np.log10(Pxx)
        mn = np.min(Pxx_den_db)
        mx = np.max(Pxx_den_db)
        file_object = open(log_file, "a")
        file_object.write(f"\nfilename: {filename} \nchannel: {c} \nmin: {mn} \nmax: {mx} \n")
        file_object.close()
        del _, Pxx, mn, mx
        gc.collect()