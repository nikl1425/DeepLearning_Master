from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Input, concatenate, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet152, VGG16
from datetime import datetime, date


def get_covn_base(input_layer, img_shape):
    input = Conv2D(64, (3,3), input_shape=img_shape)(input_layer)
    acti01 = Activation('relu')(input)
    pool01 = MaxPooling2D((2, 2))(acti01)
    covn02 = Conv2D(32, (3, 3), kernel_regularizer=l2(1e-5))(pool01)
    acti02 = Activation('relu')(covn02)
    pool02 = MaxPooling2D(2, 2)(acti02)
    covn03 = Conv2D(32, (3, 3), kernel_regularizer=l2(1e-5))(pool02)
    acti03 = Activation('relu')(covn03)
    covn_base = MaxPooling2D(pool_size=(2,2), padding='same')(acti03)
    return covn_base

def get_small_covn_base(input_layer, img_shape):
    covn01 = Conv2D(32, (5,5), padding='same', kernel_regularizer=l2(1e-5), input_shape=img_shape)(input_layer)
    acti01 = Activation('relu')(covn01)
    pool01 = MaxPooling2D((3, 3))(acti01)
    covn02 = Conv2D(16, (5, 5), kernel_regularizer=l2(1e-5))(pool01)
    acti02 = Activation('relu')(covn02)
    covn_base = MaxPooling2D(3, 3)(acti02)
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

def get_shallow_three_input_cnn(img_shape):

    model_one_input = Input(shape=img_shape)
    model_one = get_covn_base(model_one_input, img_shape)

    model_two_input = Input(shape=img_shape)
    model_two = get_covn_base(model_two_input, img_shape)

    model_three_input = Input(shape=img_shape)
    model_three = get_covn_base(model_two_input, img_shape)

    concat_feature_layer = concatenate([model_one, model_two, model_three])
    flatten_layer = Flatten()(concat_feature_layer)
    fully_connected_dense_big = Dense(256, activation='relu')(flatten_layer)
    dropout_one = Dropout(0.3)(fully_connected_dense_big)
    fully_connected_dense_small = Dense(128, activation='relu')(dropout_one)
    output = Dense(3, activation='softmax')(fully_connected_dense_small)

    model = Model(
    inputs=[model_one_input, model_two_input, model_three_input],
    outputs=output
    )

    return model


def get_small_three_input_cnn(img_shape):

    model_one_input = Input(shape=img_shape)
    model_one = get_small_covn_base(model_one_input, img_shape)

    model_two_input = Input(shape=img_shape)
    model_two = get_small_covn_base(model_two_input, img_shape)

    model_three_input = Input(shape=img_shape)
    model_three = get_small_covn_base(model_two_input, img_shape)

    concat_feature_layer = concatenate([model_one, model_two, model_three])
    flatten_layer = Flatten()(concat_feature_layer)
    fully_connected_dense_big = Dense(256, activation='relu')(flatten_layer)
    dropout_one = Dropout(0.3)(fully_connected_dense_big)
    fully_connected_dense_small = Dense(128, activation='relu')(dropout_one)
    output = Dense(3, activation='softmax')(fully_connected_dense_small)

    model = Model(
    inputs=[model_one_input, model_two_input, model_three_input],
    outputs=output
    )

    return model

def get_resnet(img_shape, trainable=False):

    resnet152 = ResNet152(
        weights='imagenet',
        include_top=False,
        input_shape=img_shape
    )

    for layer in resnet152.layers:
        layer.trainable = trainable
    
    print(f"Loaded resnet152.")

    return resnet152

def get_vgg16(img_shape, trainable=False):
    vgg16 = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=img_shape
    )

    for layer in vgg16.layers:
        layer.trainable = trainable

    print(f"Loaded vgg16.")

    return vgg16

def get_vgg16_resnet152(img_shape, trainable=False):

    resnet = get_resnet(img_shape, trainable)
    vgg16 = get_vgg16(img_shape, trainable)
    concat_feature_layer = concatenate([resnet.output, vgg16.output])
    fully_connected_dense_big = Dense(1024, activation='relu')(concat_feature_layer)
    dropout_one = Dropout(0.5)(fully_connected_dense_big)
    flatten_layer = Flatten()(dropout_one)
    fully_connected_dense_small = Dense(512, activation='relu')(flatten_layer)
    dropout_two = Dropout(0.5)(fully_connected_dense_small)
    output = Dense(3, activation='softmax')(dropout_two)

    model = Model(
        inputs=[resnet.input, vgg16.input],
        outputs=output
    )

    return model


def reduce_lr():
    return ReduceLROnPlateau(monitor='val_loss', 
                            factor=0.2,
                            patience=2, 
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


