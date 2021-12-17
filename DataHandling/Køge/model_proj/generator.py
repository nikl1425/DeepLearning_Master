from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf

from util import check_invalid_files, inspect_class_distribution, get_lowest_distr, limit_data, shuffle_order_dataframes


class custom_generator_three_input(tf.keras.utils.Sequence):
    """
    Custom Generator (Extends Tensorflow Sequence for multi processing).
    If shuffle = True: after each epoch generate a new dataframe of png paths.
    Pass dict class distribution if adjust class sampling.
    Generate path df upon init.
    """
    def __init__(self, ecg_path, eeg_1_path, eeg_2_path, batch_size, img_shape, class_distribution = {}, shuffle=True, X_col='filename', Y_col='class'):
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
        if shuffle:
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

    def __get_data(self, x1_batches):
        eeg_1_path_batch = x1_batches[self.X_col]
        eeg_2_path_batch = x1_batches[self.X_col]
        ecg_path_batch = x1_batches[self.X_col]

        label_batch = x1_batches[self.Y_col]

        x1_batch = np.asarray([self.__get_input(x, self.img_shape) for x in eeg_1_path_batch])
        x2_batch = np.asarray([self.__get_input(x, self.img_shape) for x in eeg_2_path_batch])
        x3_batch = np.asarray([self.__get_input(x, self.img_shape) for x in ecg_path_batch])
        y_batch = np.asarray([self.__get_output(y, self.n_name) for y in label_batch])

        return tuple([x1_batch, x2_batch, x3_batch]), y_batch

    def __getitem__(self, index):
        n_batches = self.eeg_1_df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(n_batches)        
        return X, y

    def __len__(self):
        return self.len // self.batch_size


generator = ImageDataGenerator(rescale = 1./255)

def create_data_generator(data_gen_one, data_gen_two):
    """
    Base generator two input model.
    """
    while(True):
        _gen1, _gen1_l = next(data_gen_one)
        _gen2, _gen2_l = next(data_gen_two)

        yield [_gen1, _gen2], np.array(_gen1_l)

def create_batch_generator(df_a, df_b, img_shape, batch_size=10):
    input_1_train_gen = generator.flow_from_dataframe(
        df_a,
        batch_size=batch_size, 
        target_size=img_shape, 
        shuffle=False,
        color_mode="rgb",
        class_mode="categorical",
        subset="training")

    input_2_train_gen = generator.flow_from_dataframe(
        df_b,
        batch_size=batch_size, 
        target_size=img_shape, 
        shuffle=False,
        color_mode="rgb",
        class_mode="categorical",
        subset="training")

    input_1_validation_gen = generator.flow_from_dataframe(
        df_a,
        batch_size=batch_size, 
        target_size=img_shape, 
        shuffle=False,
        color_mode="rgb",
        class_mode="categorical",
        subset="validation")


    input_2_validation_gen = generator.flow_from_dataframe(
        df_b,
        batch_size=batch_size, 
        target_size=img_shape, 
        shuffle=False,
        color_mode="rgb",
        class_mode="categorical",
        subset="validation")

    multi_train_generator = create_data_generator(
        input_1_train_gen,
        input_2_train_gen
    )

    multi_validation_generator = create_data_generator(
        input_1_validation_gen,
        input_2_validation_gen
    )

    train_samples = input_1_train_gen.samples
    val_samples = input_1_validation_gen.samples

    return multi_train_generator, multi_validation_generator, train_samples, val_samples

def test_generator(ecg_path, eeg_1_path, eeg_2_path, img_shape, batch_size=1):
    """
    Test generator.
    Batch = 1
    Shuffle default False.
    Categorical label
    """

    test_gen1 = generator.flow_from_directory(
        ecg_path,
        batch_size=batch_size, 
        target_size=img_shape, 
        shuffle=False,
        color_mode="rgb",
        class_mode="categorical")

    test_gen2 = generator.flow_from_dataframe(
        eeg_1_path,
        batch_size=1, 
        target_size=img_shape, 
        shuffle=False,
        color_mode="rgb",
        class_mode="categorical")

    test_gen3 = generator.flow_from_dataframe(
        eeg_1_path,
        batch_size=1, 
        target_size=img_shape, 
        shuffle=False,
        color_mode="rgb",
        class_mode="categorical")
    
    test_steps = test_gen1.samples // batch_size

    multi_test_generator = create_data_generator(
        test_gen1,
        test_gen2,
        test_gen3
    )

    # print and validate identical y_true label
    [custom_print(test_gen1.filenames[i], i, x) for i, x in enumerate(test_gen1.classes[0:5])]

    [custom_print(test_gen2.filenames[i], i, x) for i, x in enumerate(test_gen2.classes[0:5])]

    [custom_print(test_gen3.filenames[i], i, x) for i, x in enumerate(test_gen2.classes[0:5])]

    y_true = test_gen1.classes

    

    return multi_test_generator, test_steps, y_true




def custom_print(f, i, x):
    print(f"index : {i}, y_true : {x} : filename : {f}")

if __name__ == "__main__":
    class_mapping = {"sz": 1, "non-sz": 0}
    x = tf.keras.utils.to_categorical(class_mapping["non-sz"], num_classes=2)
    print(x)