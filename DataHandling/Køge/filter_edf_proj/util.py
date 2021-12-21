from datetime import datetime
from matplotlib.pyplot import sca
import pandas as pd
import numpy as np
from scipy.signal import welch
import gc
import matplotlib.pyplot as plt

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

def find_log_min_max_welch(channel, signal,  filename, log_file, freq=500):
    '''
    Calculate Power median of the signal.
    Mask the PSD if filters creates low power.
    Then log it to txt and use for pcolormesh normalization.
    '''

    # PSD w. Welch
    f, Pxx = welch(signal[channel], fs=freq, window='hanning', scaling='density', average='median', detrend=False)
    Pxx_den_db = nanpow2db(Pxx)

    # Creating and remove frequencies === False e.g. not in mask

    # mask for low pass filter
    mask = (f < 240)
    m_Pxx_den_db = Pxx_den_db[mask]
    m_f = f[mask]

    # mask for bandpass
    mask = (m_f < 47) | (m_f > 53)
    m_Pxx_den_db = m_Pxx_den_db[mask]
    m_f = m_f[mask]
    
    # mask for bandpass
    mask = (m_f < 97) | (m_f > 103)
    m_Pxx_den_db = m_Pxx_den_db[mask]
    m_f = m_f[mask]

    # mask for highpass
    mask = (m_f > 1)
    m_Pxx_den_db = m_Pxx_den_db[mask]
    m_f = m_f[mask]

    # Debugging only:
    # plt.plot(m_f, m_Pxx_den_db)
    # plt.savefig("hi.png")

    # Filtered and masked global median min and max value
    mn = np.min(m_Pxx_den_db)
    mx = np.max(m_Pxx_den_db)

    # Log file name and value pair
    file_object = open(log_file, "a")
    file_object.write(f"\nfilename: {filename} \nchannel: {channel} \nmin: {mn} \nmax: {mx} \n")
    file_object.close()

    # Garbage collection
    del f, Pxx, mn, mx, mask, m_Pxx_den_db, m_f
    gc.collect()

def nanpow2db(y):
    if isinstance(y, int) or isinstance(y, float):
        if y == 0:
            return np.nan
        else:
            ydB = 10 * np.log10(y)
    else:
        if isinstance(y, list):  # if list, turn into array
            y = np.asarray(y)
        y = y.astype(float)  # make sure it's a float array so we can put nans in it
        y[y == 0] = np.nan
        ydB = 10 * np.log10(y)

    return ydB