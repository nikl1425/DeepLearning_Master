# Module + packages
import sys
import os
from sklearn.utils import shuffle
import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import os
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# Helpers:
from util import create_validation_dir, remove_DSSTORE
from model import get_shallow_cnn, get_vgg16_resnet152, reduce_lr, checkpoint, save_history, save_model
from generator import create_batch_generator, test_generator, custom_generator_three_input
from plot import plot_con_matrix, evaluate_training_plot


# Ubuntu Path Route:
# external_hdd_path = "/media/deepm/NHR HDD/"
# external_proj_path = "Køge_04/"
# os.chdir(external_hdd_path + external_proj_path)

# Mac Path Route:
external_hdd_path = "/Volumes/NHR HDD/"
external_proj_path = "Køge_04/"
os.chdir(external_hdd_path + external_proj_path)
print(os.getcwd())

# Folders
png_path = "Windows/"
train_path = "train/"
validation_path = "validation/"
eeg_all_freq_png = "EEG_ALL_FREQ/"
eeg_low_freq_png = "EEG_LOW_FREQ/"
ecg_png = "ECG/"
eval_path = "model_evaluation/"

# Training sets:
eeg_all_train_path = f"{png_path}{train_path}{eeg_all_freq_png}"
eeg_low_train_path = f"{png_path}{train_path}{eeg_low_freq_png}"
ecg_train_path = f"{png_path}{train_path}{ecg_png}"

# Validation sets:
eeg_all_validation_path = f"{png_path}{validation_path}{eeg_all_freq_png}"
eeg_low_validation_path = f"{png_path}{validation_path}{eeg_low_freq_png}"
ecg_validation_path = f"{png_path}{validation_path}{ecg_png}"

# Training Config:
epoch_train = 20
b_size = 20
start_lr = 0.0005
loss_function = "categorical_crossentropy"
train_class_distribution = {"Interictal": 3, "Preictal": 3, "Seizure": 1}


def get_img_input_shape(for_model=False):
    if for_model:
        return(224,224,3)
    return (224, 224)

def run():
    remove_DSSTORE(eeg_all_train_path)
    remove_DSSTORE(eeg_low_train_path)
    remove_DSSTORE(ecg_train_path)



    # Shift OS check files:
    #check_invalid_files(eeg_img_path)
    #check_invalid_files(ecg_img_path)

    model = get_vgg16_resnet152(get_img_input_shape(True), trainable=True)

    print(model.summary())

    # tf.keras.utils.plot_model(
    #                     model,
    #                     to_file="model.png",
    #                     show_shapes=False,
    #                     show_dtype=False,
    #                     show_layer_names=True,
    #                     rankdir="TB",
    #                     expand_nested=False,
    #                     dpi=96,
    # )

    opt = Adam(learning_rate=start_lr)

    model.compile(
        loss=loss_function,
        optimizer=opt,
        metrics=[tensorflow.keras.metrics.CategoricalAccuracy()]
    )

    '''
    input A: ECG
    input B: EEG all frequency
    input C: EEG low frequency
    '''

    train_gen = custom_generator_three_input(ecg_path=ecg_train_path,
                                            eeg_1_path=eeg_all_train_path,
                                            eeg_2_path= eeg_low_train_path,
                                            batch_size=b_size,
                                            img_shape=get_img_input_shape(for_model=True),
                                            shuffle=True)
    validation_gen = custom_generator_three_input(ecg_path=ecg_validation_path,
                                                eeg_1_path=eeg_all_validation_path,
                                                eeg_2_path= eeg_low_validation_path,
                                                batch_size=b_size,
                                                img_shape=get_img_input_shape(for_model=True),
                                                shuffle=False)

    history = model.fit(
        train_gen,
        epochs=epoch_train,
        validation_data=validation_gen,
        callbacks=[checkpoint(), reduce_lr()]
    )

    save_history(eval_path + "history.txt", history.history)
    save_model(model, eval_path, "cnn")
    evaluate_training_plot(history, eval_path, same_plot=True)
    evaluate_training_plot(history, eval_path, same_plot=False)

    test_gen, test_step, y_true = test_generator(ecg_path=ecg_validation_path, 
                                                eeg_1_path=eeg_all_validation_path,
                                                eeg_2_path=eeg_low_validation_path,
                                                img_shape=get_img_input_shape())
    pred = model.predict(test_gen, steps=test_step)
    y_pred = pred.argmax(axis=-1)
    labels = ["Seizure", "Preictal", "Interictal"]

    clf_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)

    print(clf_report)

    confusion_matx = confusion_matrix(y_true=y_true, y_pred=list(y_pred), normalize='all')

    plot_con_matrix(confusion_matx, eval_path, labels)

if __name__ == "__main__":
    # Creating validation set: Default params = 20% of .png in dir move to val dir

    '''
    Only run create_validation_dir:
    if no validation  split is created.
    If in need of test split etc.
    '''
    #create_validation_dir(eeg_all_train_path, eeg_all_validation_path)

    '''
    Train and evaluate model.
    '''
    #run()

    train_gen = custom_generator_three_input(ecg_path=ecg_train_path,
                                            eeg_1_path=eeg_all_train_path,
                                            eeg_2_path= eeg_low_train_path,
                                            batch_size=b_size,
                                            img_shape=get_img_input_shape(for_model=True),
                                            shuffle=True,
                                            class_distribution=train_class_distribution)
    # validation_gen = custom_generator_three_input(ecg_path=ecg_validation_path,
    #                                             eeg_1_path=eeg_all_validation_path,
    #                                             eeg_2_path= eeg_low_validation_path,
    #                                             batch_size=b_size,
    #                                             img_shape=get_img_input_shape(for_model=True),
    #                                             shuffle=False,
    #                                             class_distribution=train_class_distribution)

    X, y = train_gen[0]
    X2, y2 = train_gen[0]
    #print(X)
