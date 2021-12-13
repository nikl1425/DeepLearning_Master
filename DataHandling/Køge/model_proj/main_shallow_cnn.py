
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
from generator import create_batch_generator, test_generator, custom_generator
from plot import plot_con_matrix, evaluate_training_plot


external_hdd_path = "/Volumes/NHR HDD/Køge_02/"
os.chdir(external_hdd_path)
print(os.getcwd())

eeg_img_path =  "Windows/EEG/Images/"
ecg_img_path = "Windows/EKG/Images/"
val_eeg_img_path = "Windows/EEG/Images/validation/"
val_ecg_img_path = "Windows/ECG/Images/validation/"
b_size = 5
eval_path = "shallow_eval/"

def get_img_input_shape(for_model=False):
    if for_model:
        return(299,299,3)
    return (299, 299)

def run():
    remove_DSSTORE(eeg_img_path)

    create_validation_dir(eeg_img_path, val_eeg_img_path)
    create_validation_dir(ecg_img_path, val_ecg_img_path)

    #check_invalid_files(eeg_img_path)
    #check_invalid_files(ecg_img_path)

    model = get_shallow_cnn(get_img_input_shape(True))

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

    train_gen = custom_generator(ecg_path=ecg_img_path,
                                eeg_path=eeg_img_path,
                                batch_size=b_size,
                                img_shape=get_img_input_shape(for_model=True),
                                shuffle=True)

    history = model.fit(
        train_gen,
        epochs=2,
        callbacks=[checkpoint(), reduce_lr()]
    )

    save_history(eval_path + "history.txt", history.history)
    save_model(model, eval_path, "shallow_cnn")
    evaluate_training_plot(history, eval_path, same_plot=True)
    evaluate_training_plot(history, eval_path, same_plot=False)

    test_gen, test_step, y_true = test_generator(val_ecg_img_path, 
                                                 val_eeg_img_path,
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