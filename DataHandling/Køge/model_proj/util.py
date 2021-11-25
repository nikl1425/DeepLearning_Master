import PIL
from pathlib import Path
import os
import random
import pandas as pd
import numpy as np

def check_invalid_files(img_path):
    print("check for invalid files")
    path = Path(img_path).rglob("*.png")
    for img_p in path:
        try:
            img = PIL.Image.open(img_p)
        except PIL.UnidentifiedImageError:
                print(f"FOUND NON VALID IMAGE | path : {img_p} ")

def inspect_class_distribution(path):
    dist_list = {}
    for i in os.listdir(path):
        classname = i
        number_of_png = str(len([x for x in os.listdir(path + "/" + i)]))
        #dict = {classname, number_of_png}
        dist_list[classname] =  number_of_png
    return dist_list

def get_lowest_distr(dict_a, dict_b):

    concat_dict = []
    concat_dict.append(dict_a)
    concat_dict.append(dict_b)
    min_val = 0

    for item in concat_dict:
        for key in item:
            try:
                item[key] = int(item[key])
            except ValueError:
                item[key] = float(item[key])
            if min_val == 0:
                min_val = item[key]
            elif item[key] < min_val:
                min_val = item[key]
    return int(min_val)

def limit_data(data_dir, n=0):
    a=[]
    if n > 0:
        for i in os.listdir(data_dir):
            image_path = random.sample(os.listdir(data_dir+'/'+i), n)
            for k,j in enumerate(image_path):
                if k>n:continue
                a.append((f'{data_dir}/{i}/{j}',i))
    return pd.DataFrame(a,columns=['filename','class']).reset_index(drop=True)

def shuffle_order_dataframes(df_a, df_b):
    same_len = len(df_a) == len(df_b)
    if same_len:
        idx = np.random.permutation(df_a.index)
        df_a = df_a.reindex(idx, axis=0)
        df_b = df_b.reindex(idx, axis=0)
        print(df_a.head())
        print(df_b.head())
        return df_a, df_b
    print("not same len = not shuffled identically")
    return df_a, df_b
