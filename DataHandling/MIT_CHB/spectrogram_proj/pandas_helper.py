import pandas as pd
from datetime import datetime


def read_compressed_df(path, rel_channel):
    df = pd.read_csv(path, usecols=rel_channel)

    channels = [item for item in list(df.columns) if item != "class" if item != "timestamp" if item != "Unnamed: 0"]

    print(f"COLUMNS = {df.columns}")

    return (df, channels)


def get_window(channel, start_index, data, size = 4, overlap = 0, is_sezure = False, frequency = 256):
    if(is_sezure):
        overlap = 2
    else:
        overlap = overlap
    
    start = start_index * (size - overlap) * frequency
    end = start + (size * frequency)
    
    return [data[channel][start:end].tolist()]

def get_max_window_iteration(dataframe, buffer, FREQ = 256):
    len_of_df = int(len(dataframe) // (buffer*FREQ))
    return len_of_df


