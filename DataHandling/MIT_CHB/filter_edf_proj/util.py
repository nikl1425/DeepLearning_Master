from datetime import datetime
import pandas as pd
import numpy as np

summary_txt_file_type = "-summary.txt"
edf_file_type = ".edf"

def read_format_info_file(txt_summary_file_path):
    str_container = ""
    with open(txt_summary_file_path, 'r') as f:
        for line in f:
            str_container += str(line).replace("\n", "<br>")

    formatted_str = re.findall('(.*?)<br><br>', str_container)
    bla = [x.group() for x in re.finditer('(.*?)<br><br>', str_container)]

    formatted_str = [x for x in formatted_str if "File Name" in x]

    return formatted_str

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

def get_all_patient_folder_names(database_path):
    folders = os.listdir(database_path)
    patient_folder_names = [(database_path + "/" + x) for x in folders if (x.find(".DS_Store") == -1) if not "._" in x]

    return patient_folder_names

def get_all_file_names(directory):
    files = os.listdir(directory)
    edfFileNameList = [(directory + "/" + x) for x in files if (x.endswith(edf_file_type)) if not "._" in x]
    summary_info_file_name = [(directory + "/" + x) for x in files if (x.endswith(summary_txt_file_type))]
    return (summary_info_file_name[0], edfFileNameList)


class FileInformationContainer:
    def __init__(self, information_str):
        self.information_str = self.clean_string(information_str)
        self.file_name = self.set_filename()
        self.time_start = self.set_file_time_start_ms()
        self.sz_info = self.set_sz_information()
        
    def clean_string(self, uncleaned_str):
        return uncleaned_str.replace("<br>", " ")

    def set_filename(self):
        filename_found = re.match(r"^File Name: (.+?).edf", self.information_str)
        if filename_found:
            return filename_found.group(1)
        else:
            print(f"{self.file_name} failed get_filename")
            return "filename not found"
    
    def get_milli_sec(self, time_str):
        """Get Seconds from time."""
        dt_obj = datetime.strptime(time_str,'%H:%M:%S')
        millisec = dt_obj.timestamp() * 1000
        return millisec

    def set_file_time_start_ms(self):
        time_start_found = re.match(r".*File Start Time: (.*?) File", self.information_str)
        if time_start_found:
            try:
                return self.get_milli_sec(time_start_found.group(1))
            except Exception as e :
                print(f"{self.file_name}: error {e} cannot convert to ms time")
                return f"{e}"
        else:
            print(f"{self.file_name} failed get_file_time_start_ms")
            return "time start not found"
    
    def get_sz_count(self):
        sz_count = 0
        count_found = re.search(r".*Seizures in File: (.*?) Seizure", self.information_str)
        if count_found:
            sz_count = count_found.group(1)
        if int(sz_count) > 0:
            return int(sz_count)
        else:
            return 0
        

    def set_sz_information(self):
        if(type(self.get_sz_count()) != None and self.get_sz_count() > 0):
            try:
                pattern = re.compile(r"Seizure [1-9] (?P<state>[?:Start|End]+) Time: (?P<Sec>[0-9]+)")
                myList = [m.groupdict() for m in pattern.finditer(self.information_str)]
                if(len(myList) <= 0):
                     pattern = re.compile(r"Seizure (?P<state>[?:Start|End]+) Time: (?P<Sec>[0-9]+)")
                     myList = [m.groupdict() for m in pattern.finditer(self.information_str)]
                for item in myList:
                    converted_time = int(item.get("Sec"))
                    item["Sec"] = converted_time
                formatted = []
                for i in range(0, len(myList), 2):
                    formatted.append({"sz_start" : myList[i]["Sec"], "sz_end" : myList[i+1]["Sec"]})
                return formatted
            except Exception as e:
                print(f"set_sz_information failed at file: {self.file_name} with the following exception: {e}")
        else:
            return []

    def get(self):
        return self.information_str


def sort_remove_files(list_obj):
  new_lst = []
  for x in range(0, len(list_obj)):
    if x+1 < len(list_obj) and FileInformationContainer(list_obj[x + 1]).get_sz_count() > 0 and FileInformationContainer(list_obj[x]).get_sz_count() == 0:
        new_lst.append(list_obj[x])
    if FileInformationContainer(list_obj[x]).get_sz_count() > 0:
      new_lst.append(list_obj[x])
  return new_lst