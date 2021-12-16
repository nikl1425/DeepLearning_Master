import pandas as pd
from datetime import datetime


def read_compressed_df(path, rel_channel):
    df = pd.read_csv(path, usecols=rel_channel)

    

    channels = [item for item in list(df.columns) if item != "class" if item != "timestamp" if item != "Unnamed: 0"]

    print(f"COLUMNS = {df.columns}")

    return (df, channels)


def get_window(channel, start_index, data, size = 4, overlap = 0, is_sezure = False, frequency = 500):
    if(is_sezure):
        overlap = 2
    else:
        overlap = overlap
    
    start = start_index * (size - overlap) * frequency
    end = start + (size * frequency)

    date_timestamp = ""

    try:
        date_timestamp = datetime.fromtimestamp(s(data['timestamp'][start:start+1]/1000).tolist()[0]).strftime('%H:%M:%S')
    except:
        date_timestamp = "datetime cannot be converted"

    print(f"Window: {start_index}")
    
    return [data[channel][start:end].tolist(), date_timestamp]

def get_max_window_iteration(dataframe, buffer, FREQ = 500):
    len_of_df = int(len(dataframe) // (buffer*FREQ))
    return len_of_df


