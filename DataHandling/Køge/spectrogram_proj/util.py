import re


def logging_info_txt(string_to_write, path):
    file_object = open(path, "a")
    file_object.write(string_to_write)
    file_object.close()

def find_frq(str_obj):
    frq_found = re.search(r"<br> freq: (.*?) <br>", str_obj)
    if(frq_found):
        return int(float(frq_found.group(1)))

def find_filename(str_obj):
    frq_found = re.search(r"filename: (.*?) <br>", str_obj)
    if(frq_found):
        return (frq_found.group(1))

def find_welch_max(str_obj):
    max_found = re.search(r"max: (.*?) ", str_obj)
    if(max_found):
        return (max_found.group(1))

def find_welch_min(str_obj):
    max_found = re.search(r"min: (.*?) <br>", str_obj)
    if(max_found):
        return (max_found.group(1))

def find_channel(str_obj):
    
    ch_found = re.search(r"channels: \[(.*?)\] ", str_obj)
    if(ch_found):
        print("FOUND")
        print(ch_found.group(1))
        channels = [x.replace("'", "").strip() for x in list(ch_found.group(1).split(","))] 
        return channels
        

def get_info_text(path):
    with open(path, 'r') as f:
        str_container = ""
        formatted_str = []
        for line in f:
            #print(line)
            str_container += str(line).replace("\n", "<br>")
        #print(str_container)
        formatted_str = re.findall('(.*?)<br><br>', str_container)
        return formatted_str