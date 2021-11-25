
# Module + packages
import sys
import os
import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import os
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
from util import check_invalid_files, inspect_class_distribution, get_lowest_distr, limit_data, shuffle_order_dataframes
from model import get_shallow_cnn, get_vgg16_resnet152, reduce_lr, checkpoint, save_history, save_model
from generator import create_batch_generator, create_test_generator
from plot import plot_con_matrix, evaluate_training_plot


external_hdd_path = "/Volumes/NHR HDD/Køge_02/"
os.chdir(external_hdd_path)
print(os.getcwd())
eeg_img_path =  "Windows/EEG/Images/"
ecg_img_path = "Windows/EKG/Images/"
batch_size = 18
eval_path = "shallow_eval/"

def get_img_input_shape(for_model=False):
    if for_model:
        return(299,299,3)
    return (299, 299)

def run():

    try:
        os.remove(eeg_img_path + "/.DS_Store")
    except FileNotFoundError as e:
        print(f"file not found with error: {e}")

    #check_invalid_files(eeg_img_path)
    #check_invalid_files(ecg_img_path)

    eeg_class_dist = inspect_class_distribution(eeg_img_path)
    ecg_class_dist = inspect_class_distribution(ecg_img_path)
    print(f"eeg distribution: {eeg_class_dist} \n ecg distribution: {ecg_class_dist}")

    print(f"Lowest distributed class: {get_lowest_distr(eeg_class_dist, ecg_class_dist)}")
    max_n_images = get_lowest_distr(ecg_class_dist, eeg_class_dist)

    balanced_ecg_data = limit_data(ecg_img_path, n=max_n_images).sort_values(by=['class']).reset_index(drop=True)
    balanced_eeg_data = limit_data(eeg_img_path, n=max_n_images).sort_values(by=['class']).reset_index(drop=True)

    print(f"balance ecg df: {balanced_ecg_data['class'].value_counts()}")
    print(f"balance eeg df: {balanced_eeg_data['class'].value_counts()}")

    balanced_eeg_data, balanced_ecg_data = shuffle_order_dataframes(balanced_eeg_data, balanced_ecg_data)

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

    opt = Adam()

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=[tensorflow.keras.metrics.CategoricalAccuracy()]
    )

    train_gen, val_gen, train_sam, val_sam = create_batch_generator(
        balanced_eeg_data,
        balanced_ecg_data,
        get_img_input_shape(),
        batch_size)

    history = model.fit(
        train_gen,
        epochs=2,
        steps_per_epoch = train_sam//batch_size, 
        validation_data=val_gen, 
        validation_steps = val_sam//batch_size,
        callbacks=[checkpoint(), reduce_lr()]
    )

    save_history(eval_path + "history.txt", history.history)
    save_model(model, eval_path, "shallow_cnn")
    evaluate_training_plot(history, eval_path, same_plot=True)
    evaluate_training_plot(history, eval_path, same_plot=False)

    test_gen, test_step, y_true = create_test_generator(balanced_eeg_data,
                                                        balanced_ecg_data,
                                                        get_img_input_shape())
    pred = model.predict(test_gen, steps=test_step)
    y_pred = pred.argmax(axis=-1)
    labels = ['Interictal', 'Preictal', 'Seizure']

    clf_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)

    print(clf_report)

    confusion_matx = confusion_matrix(y_true=y_true, y_pred=list(y_pred), normalize='all')

    plot_con_matrix(confusion_matx, eval_path, labels)

if __name__ == "__main__":
    run()