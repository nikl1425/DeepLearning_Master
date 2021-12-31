# Module + packages
import sys
import os
from sklearn.utils import shuffle
import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
tf.config.run_functions_eagerly(False)
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# Helpers:
from util import create_validation_dir, create_test_dir, remove_DSSTORE
from model import get_shallow_cnn, get_vgg16_resnet152, get_shallow_three_input_cnn, reduce_lr, checkpoint, save_history, save_model, load_saved_model, get_small_three_input_cnn
from generator import custom_generator_three_input, test_generator_three_input
from plot import plot_con_matrix, evaluate_training_plot, show_batch


# Ubuntu Path Route:
# external_hdd_path = "/media/deepm/NHR HDD/"
# external_proj_path = "Køge_04/"
#os.chdir(external_hdd_path + external_proj_path)

# Windows Path Route:
# external_hdd_path = "E:/"
# external_proj_path = "Eks_DB/"
# os.chdir(external_hdd_path + external_proj_path)

# Mac Path Route:
external_hdd_path = "/Volumes/NHR HDD/"
external_proj_path = "Eks_DB/"
os.chdir(external_hdd_path + external_proj_path)
print("CURRENT DIR: \n" + os.getcwd())

# Folders
png_path = "Køge/spectrogram/"
train_path = "train/"
validation_path = "validation/"
test_path = "test/"
eeg_all_freq_png = "EEG_ALL/"
eeg_low_freq_png = "EEG_LOW/"
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


# Test sets:
eeg_all_test_path = f"{png_path}{test_path}{eeg_all_freq_png}"
eeg_low_test_path = f"{png_path}{test_path}{eeg_low_freq_png}"
ecg_test_path = f"{png_path}{test_path}{ecg_png}"

# Training Config:
epoch_train = 10
b_size = 18
start_lr = 0.00001
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

    #model = get_vgg16_resnet152(get_img_input_shape(True))
    #model = tensorflow.keras.models.load_model("Shallow_checkpoint.h5")
    #model = load_saved_model("/media/deepm/NHR HDD/Køge_04/model_evaluation/cnn.h5")
    model = get_shallow_three_input_cnn(img_shape=get_img_input_shape(True))

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
    Generator custom input:
    input 1: ECG
    input B: EEG all frequency
    input C: EEG low frequency

    output in tuples inside batches of 32:
    [eeg_all, eeg_low, ecg]

    Model inputs:
    1st. input = EEG_ALL
    2nd. input = EEG_LOW
    3rd. input = ECG
    '''

    train_gen = custom_generator_three_input(ecg_path=ecg_train_path,
                                            eeg_1_path=eeg_all_train_path,
                                            eeg_2_path= eeg_low_train_path,
                                            batch_size=b_size,
                                            img_shape=get_img_input_shape(for_model=True),
                                            shuffle=True,
                                            class_distribution=train_class_distribution)
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

    test_gen, test_step, y_true = test_generator_three_input(ecg_path=ecg_validation_path, 
                                                            eeg_1_path=eeg_all_validation_path,
                                                            eeg_2_path=eeg_low_validation_path,
                                                            img_shape=get_img_input_shape(),
                                                            batch_size=1)
                                                                
    pred = model.predict(test_gen, steps=test_step)
    y_pred = pred.argmax(axis=-1)
    labels = ["Seizure", "Preictal", "Interictal"]

    clf_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)

    print(clf_report)

    confusion_matx = confusion_matrix(y_true=y_true, y_pred=list(y_pred), normalize='all')

    plot_con_matrix(confusion_matx, eval_path, labels)

if __name__ == "__main__":

    run()