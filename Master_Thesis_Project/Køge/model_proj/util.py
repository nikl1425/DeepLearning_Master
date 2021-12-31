import PIL
from pathlib import Path
import os
import random
import pandas as pd
import numpy as np
import shutil
import glob


def check_invalid_files(img_path):
    """
    Loop through folder and print invalid png.
    """
    print("check for invalid files")
    path = Path(img_path).rglob("*.png")
    for img_p in path:
        try:
            img = PIL.Image.open(img_p)
        except PIL.UnidentifiedImageError:
                print(f"FOUND NON VALID IMAGE | path : {img_p} ")

def inspect_class_distribution(path):
    """
    Give path.
    Inspect n file in each subfolder.
    """

    dist_list = {}
    for i in os.listdir(path):
        classname = i
        number_of_png = str(len([x for x in os.listdir(path + "/" + i)]))
        #dict = {classname, number_of_png}
        dist_list[classname] =  number_of_png
    return dist_list

def get_lowest_distr(dict_a, dict_b, dict_c=None):
    """
    Multi Input model requires identical batches.
    Pass two dict and it finds the lowest distributes class.
    """

    concat_dict = []      
    min_val = 0
    concat_dict.append(dict_a)
    concat_dict.append(dict_b)

    if dict_c is not None:
        concat_dict.append(dict_c)

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

def limit_data(data_dir, n=0, config = {}):
    """
    Looks in subfolders of data_dir.
    Select n random files.
    Saves full path to file in dataframe.
    For each row save parent dir as class in dataframe.
    """
    a=[]
    if n > 0:
        mylist = os.listdir(data_dir)
        if config and isinstance(config, dict): 
            try:
                for i in os.listdir(data_dir):
                    check_k = "" if len([key for key, _ in config.items() if key.lower() in i.lower()]) == 0 else [key for key, val in config.items() if key.lower() in i.lower()][0]
                    if check_k != "":
                        n_count = config[check_k] * n if config[check_k] > 0 and check_k in i else n
                        image_path = random.sample(os.listdir(data_dir+'/'+i), n_count)
                        for k,j in enumerate(image_path):
                            if k>n_count :continue
                            a.append((f'{data_dir}/{i}/{j}',i))
                return pd.DataFrame(a,columns=['filename','class']).reset_index(drop=True)
            except:
                print("The limit dat function has the wrong time and element. Implement key=string, value=int")

        for i in os.listdir(data_dir):
            image_path = random.sample(os.listdir(data_dir+'/'+i), n)
            for k,j in enumerate(image_path):
                if k>n:continue
                a.append((f'{data_dir}/{i}/{j}',i))
        return pd.DataFrame(a,columns=['filename','class']).reset_index(drop=True)
        


def shuffle_order_dataframes(df_a, df_b, df_c=None, testing=True):
    """
    Takes two dataframe.
    Shuffle both based on index in 1st. dataframe
    """

    same_len = False
    df_c_passed = False

    if df_c is not None:
        df_c_passed = True
        same_len = len(df_a) == len(df_b) == len(df_c)
    else:
        same_len = len(df_a) == len(df_b)
    if same_len:
        idx = np.random.permutation(df_a.index)
        df_a = df_a.reindex(idx, axis=0)
        df_b = df_b.reindex(idx, axis=0)
        if df_c_passed:
            df_c = df_c.reindex(idx, axis=0)
    else:
        print("not same len = not shuffled identically")

        if testing:
            print(df_a.head())
            print(df_b.head())
            print(df_c.head())
    if df_c_passed:
        return df_a, df_b, df_c    
    return df_a, df_b

def remove_DSSTORE(path):
    """
    Removal of .DS_STORE
    """
    try:
        os.remove(path+ "/.DS_Store")
    except FileNotFoundError as e:
        print(f"file not found with error: {e}")


def create_validation_dir(data_dir, dest_dir, validation_split=0.25):
    """
    Initialize this function at start of model main.
    Takes 20% of data in train directory and move to selected destination.
    """
    f_create_list = ["Seizure", "Interictal", "Preictal"]

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        [os.makedirs(dest_dir + "/" + f) for f in f_create_list]


    for dir in os.listdir(data_dir):
        validation_folder_empty = True
        sub_dir = f"{data_dir}/{dir}/"
        if os.path.isdir(sub_dir):
            val_split = int(len([x for x in os.listdir(sub_dir)]) * validation_split)
            print(f"validation split, dir: {sub_dir} n_files: {len([x for x in os.listdir(sub_dir)])} val_split: {val_split} ")
            filenames = random.sample(os.listdir(sub_dir), val_split)

            if os.listdir(dest_dir + "/" + dir) == []:
                print("No files found in the directory.")
            else:
                print("Some files found in the directory.")
                validation_folder_empty = False

            if(validation_folder_empty):    
                for fname in filenames:
                    srcpath = os.path.join(sub_dir, fname)
                    #print(srcpath)

                    if f_create_list[0] in srcpath:
                        if not os.path.exists(dest_dir + "/" + f_create_list[0] + "/" + fname):
                            shutil.move(srcpath, dest_dir + "/" + f_create_list[0])

                    if f_create_list[1] in srcpath:
                        if not os.path.exists(dest_dir + "/" + f_create_list[1] + "/" + fname):
                            shutil.move(srcpath, dest_dir + "/" + f_create_list[1])

                    if f_create_list[2] in srcpath:
                        if not os.path.exists(dest_dir + "/" + f_create_list[2] + "/" + fname):
                            shutil.move(srcpath, dest_dir + "/" + f_create_list[2])
                print(f"Done with {srcpath}")

def create_test_dir(data_dir, dest_dir, test_split = 0.05, max_num_dict=None):
    '''
    This function should only be run when no test set is created.
    It creates a directory with 3 classes -  if max num is specified it will instead move a number of images over into destination dir.
    '''
    f_create_list = ["Seizure", "Interictal", "Preictal"]

    if os.path.exists(dest_dir):
        print("Test directory already is created for: ")
        return None

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        [os.makedirs(dest_dir + "/" + f) for f in f_create_list]
        print("created dest dir with assosiated folders.")

        for sub_dir in os.listdir(data_dir):
            filenames = []
            num_files_to_move = 0
            if isinstance(max_num_dict, type(None)):
                num_files_to_move = int(len([x for x in os.listdir(sub_dir)]) * test_split)
                print(f"sub_dir = {sub_dir} moves : {num_files_to_move} ")
            else:
                num_files_to_move = max_num_dict[sub_dir]

            filenames = random.sample(data_dir + "/" + sub_dir + "/")

            for fname in filenames:
                srcpath = os.path.join(sub_dir, fname)

                if f_create_list[0] in srcpath:
                    if not os.path.exists(dest_dir + "/" + f_create_list[0] + "/" + fname):
                        shutil.move(srcpath, dest_dir + "/" + f_create_list[0])

                if f_create_list[1] in srcpath:
                    if not os.path.exists(dest_dir + "/" + f_create_list[1] + "/" + fname):
                        shutil.move(srcpath, dest_dir + "/" + f_create_list[1])

                if f_create_list[2] in srcpath:
                    if not os.path.exists(dest_dir + "/" + f_create_list[2] + "/" + fname):
                        shutil.move(srcpath, dest_dir + "/" + f_create_list[2])
        
        print("Created test directory with all subdirs")

def inspect_train_samples(data_dir, split=None):
    '''
    Inspecting the number of png in each subdirectory of train directories.
    We don't need to move more files than needed if for ex. ECG is 1 channel then don't move more than that from EEG sets.
    '''
    for sub_dir in os.listdir(data_dir):
        print(f"inspecting: {sub_dir}")
        DIR = data_dir + "/" + sub_dir
        num_img_in_folder = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        print(num_img_in_folder)
        if not isinstance(split, type(None)):
            try:
                print(f"split is: {float(split)} and that gives n images to after split: {num_img_in_folder * split}")
            except Exception as e:
                print(f"exception: {e}")

def create_dataframe_test(data_dir):
    a = []
    for i in os.listdir(data_dir):
        image_path = os.listdir(data_dir+'/'+i)
        for k,j in enumerate(image_path):
            a.append((f'{data_dir}/{i}/{j}',i))
    return pd.DataFrame(a,columns=['filename','class']).reset_index(drop=True)
                
                




        

        