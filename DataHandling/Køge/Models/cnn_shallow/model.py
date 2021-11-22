from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Input, concatenate, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from datetime import datetime, date


def get_covn_base(input_layer, img_shape):
    input = Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(1e-5), input_shape=img_shape)(input_layer)
    covn01 = Conv2D(32, (3, 3), kernel_regularizer=l2(1e-5))(input)
    acti01 = Activation('relu')(covn01)
    pool01 = MaxPooling2D((2, 2))(acti01)
    covn02 = Conv2D(64, (3, 3), kernel_regularizer=l2(1e-5))(pool01)
    acti02 = Activation('relu')(covn02)
    pool02 = MaxPooling2D(2, 2)(acti02)
    covn03 = Conv2D(128, (3, 3), kernel_regularizer=l2(1e-5))(pool02)
    acti03 = Activation('relu')(covn03)
    pool03 = MaxPooling2D(pool_size=(2,2), padding='same')(acti03)
    covn_base = Dropout(0.2)(pool03)

    return covn_base

def get_shallow_cnn(img_shape):

    model_one_input = Input(shape=img_shape)
    model_one = get_covn_base(model_one_input, img_shape)

    model_two_input = Input(shape=img_shape)
    model_two = get_covn_base(model_two_input, img_shape)

    concat_feature_layer = concatenate([model_one, model_two])
    flatten_layer = Flatten()(concat_feature_layer)
    fully_connected_dense_big = Dense(256, activation='relu', kernel_regularizer=l2(1e-5))(flatten_layer)
    dropout_one = Dropout(0.3)(fully_connected_dense_big)
    fully_connected_dense_small = Dense(128, activation='relu', kernel_regularizer=l2(1e-5))(dropout_one)
    dropout_two = Dropout(0.3)(fully_connected_dense_small)
    output = Dense(3, activation='softmax')(dropout_two)

    model = Model(
    inputs=[model_one_input, model_two_input],
    outputs=output
    )

    return model

def reduce_lr():
    return ReduceLROnPlateau(monitor='val_loss', 
                            factor=0.2,
                            patience=7, 
                            min_lr=0.00001)

def checkpoint():
    return ModelCheckpoint(filepath="Shallow_checkpoint.h5",
                            monitor='val_categorical_accuracy',
                            mode='max',
                            save_best_only=True)


def save_history(path_to_file, history):
    with open(path_to_file, 'a') as f:
        f.write(str(datetime.now()) + "\n" + str(history) +"\n")
        f.close()

def save_model(model, folder_path, file_name):
    h5_format = ".h5"
    model.save(f"{folder_path}{file_name}{h5_format}")


