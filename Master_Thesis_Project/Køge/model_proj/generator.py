from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf

from util import check_invalid_files, inspect_class_distribution, get_lowest_distr, limit_data, shuffle_order_dataframes, create_dataframe_test



def custom_print(f, i, x):
    print(f"index : {i}, y_true : {x} : filename : {f}")

class custom_generator_three_input(tf.keras.utils.Sequence):
    """
    Custom Generator (Extends Tensorflow Sequence for multi processing).
    If shuffle = True: after each epoch generate a new dataframe of png paths.
    Pass dict class distribution if adjust class sampling.
    Generate path df upon init.
    """
    def __init__(self, ecg_path, eeg_1_path, eeg_2_path, batch_size, img_shape, class_distribution = {}, shuffle=True, X_col='filename', Y_col='class', is_test=False):
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.class_distribution = class_distribution
        self.shuffle = shuffle
        self.X_col = X_col
        self.Y_col = Y_col
        self.class_mapping = {"Seizure": 0, "Preictal": 1, "Interictal": 2} # ændre dette
        self.ecg_path = ecg_path
        self.eeg_1_path = eeg_1_path
        self.eeg_2_path = eeg_2_path
        self.eeg_1_df, self.eeg_2_df, self.ecg_df = self.__generate_data()
        self.len = len(self.eeg_1_df)
        self.n_name = self.ecg_df[self.Y_col].nunique()
        self.is_test = is_test
        
    def __generate_data(self):
        eeg_1_class_dist = inspect_class_distribution(self.eeg_1_path)
        eeg_2_class_dist = inspect_class_distribution(self.eeg_2_path)
        ecg_class_dist = inspect_class_distribution(self.ecg_path)
        max_n_images = get_lowest_distr(ecg_class_dist, eeg_1_class_dist, eeg_2_class_dist)
        balanced_ecg_data = limit_data(self.ecg_path, max_n_images, self.class_distribution).sort_values(by=[self.Y_col]).reset_index(drop=True)
        balanced_eeg_1_data = limit_data(self.eeg_1_path, max_n_images, self.class_distribution).sort_values(by=[self.Y_col]).reset_index(drop=True)
        balanced_eeg_2_data = limit_data(self.eeg_1_path, max_n_images, self.class_distribution).sort_values(by=[self.Y_col]).reset_index(drop=True)

        print(f"ECG\n{balanced_ecg_data['class'].value_counts()}")
        print(f"EEG ALL\n{balanced_ecg_data['class'].value_counts()}")
        print(balanced_eeg_1_data.head())
        print(f"EEG LOW\n{balanced_ecg_data['class'].value_counts()}")

        return shuffle_order_dataframes(balanced_eeg_1_data, balanced_eeg_2_data, balanced_ecg_data)

    def on_epoch_end(self):
        if self.shuffle:
            self.eeg_1_df, self.eeg_2_df, self.ecg_df = self.__generate_data()
            print("Shuffled!")
            
    def __get_input(self, path, target_size):
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()
        return image_arr/255.

    def __get_output(self, label, num_classes):
        print(self.n_name)
        categoric_label = self.class_mapping[label]
        return tf.keras.utils.to_categorical(categoric_label, num_classes=num_classes)

    def __get_data(self, x1_batches, x2_batches, x3_batches):
        ecg_path_batch = x1_batches[self.X_col]
        eeg_1_path_batch = x2_batches[self.X_col]
        eeg_2_path_batch = x3_batches[self.X_col]
    
        label_batch = x1_batches[self.Y_col]

        x1_batch = np.asarray([self.__get_input(x, self.img_shape) for x in eeg_1_path_batch])
        x2_batch = np.asarray([self.__get_input(x, self.img_shape) for x in eeg_2_path_batch])
        x3_batch = np.asarray([self.__get_input(x, self.img_shape) for x in ecg_path_batch])
        y_batch = np.asarray([self.__get_output(y, self.n_name) for y in label_batch])

        return tuple([x1_batch, x2_batch, x3_batch]), y_batch

    def __getitem__(self, index):
        ecg_batches = self.ecg_df[index * self.batch_size:(index + 1) * self.batch_size]
        eeg_1_batches = self.eeg_1_df[index * self.batch_size:(index + 1) * self.batch_size]
        eeg_2_batches = self.eeg_2_df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(eeg_1_batches, eeg_2_batches, ecg_batches)        
        return X, y

    def __len__(self):
        return self.len // self.batch_size

generator = ImageDataGenerator(rescale = 1./255)




class test_generator_three_input(tf.keras.utils.Sequence):
    '''
    For test we load all images in the three specified paths into memory as an np.array.
    We also need a custom data generator for this due to multiple inputs and specific labelling of the classes.
    All the test folders contains the same number of images so no need to check and random sample / shuffle
    '''
    def __init__(self, ecg_path, eeg_1_path, eeg_2_path, batch_size, img_shape, class_distribution = {}, shuffle=True, X_col='filename', Y_col='class', is_test=False):
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.class_distribution = class_distribution
        self.shuffle = shuffle
        self.X_col = X_col
        self.Y_col = Y_col
        self.class_mapping = {"Seizure": 0, "Preictal": 1, "Interictal": 2} # ændre dette
        self.ecg_path = ecg_path
        self.eeg_1_path = eeg_1_path
        self.eeg_2_path = eeg_2_path
        self.eeg_1_df, self.eeg_2_df, self.ecg_df = self.__generate_data()
        self.len = len(self.eeg_1_df)
        self.n_name = self.ecg_df[self.Y_col].nunique()
        self.is_test = is_test
        
    def __generate_data(self):
        balanced_ecg_data = create_dataframe_test(self.ecg_path)
        balanced_eeg_1_data = create_dataframe_test(self.eeg_1_path)
        balanced_eeg_2_data = create_dataframe_test(self.eeg_2_path)

        print(f"ECG\n{balanced_ecg_data['class'].value_counts()}")
        print(f"EEG ALL\n{balanced_ecg_data['class'].value_counts()}")
        print(balanced_eeg_1_data.head())
        print(f"EEG LOW\n{balanced_ecg_data['class'].value_counts()}")

        return shuffle_order_dataframes(balanced_eeg_1_data, balanced_eeg_2_data, balanced_ecg_data, testing=True)

    def on_epoch_end(self):
        if self.shuffle:
            self.eeg_1_df, self.eeg_2_df, self.ecg_df = self.__generate_data()
            
    def __get_input(self, path, target_size):
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()
        return image_arr/255.

    def __get_output(self, label, num_classes):
        print(self.n_name)
        categoric_label = self.class_mapping[label]
        return tf.keras.utils.to_categorical(categoric_label, num_classes=num_classes)

    def __get_data(self, x1_batches, x2_batches, x3_batches):
        ecg_path_batch = x1_batches[self.X_col]
        eeg_1_path_batch = x2_batches[self.X_col]
        eeg_2_path_batch = x3_batches[self.X_col]
    
        label_batch = x1_batches[self.Y_col]

        x1_batch = np.asarray([self.__get_input(x, self.img_shape) for x in eeg_1_path_batch])
        x2_batch = np.asarray([self.__get_input(x, self.img_shape) for x in eeg_2_path_batch])
        x3_batch = np.asarray([self.__get_input(x, self.img_shape) for x in ecg_path_batch])
        y_batch = np.asarray([self.__get_output(y, self.n_name) for y in label_batch])

        return tuple([x1_batch, x2_batch, x3_batch]), y_batch

    def __getitem__(self, index):
        ecg_batches = self.ecg_df[index * self.batch_size:(index + 1) * self.batch_size]
        eeg_1_batches = self.eeg_1_df[index * self.batch_size:(index + 1) * self.batch_size]
        eeg_2_batches = self.eeg_2_df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(eeg_1_batches, eeg_2_batches, ecg_batches)        
        return X, y

    def __len__(self):
        return self.len // self.batch_size