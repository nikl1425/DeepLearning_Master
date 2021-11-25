from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

generator = ImageDataGenerator(
rescale = 1./255, 
validation_split=0.2
)

def custom_sequence_dual_gen():
    pass

def create_data_generator(data_gen_one, data_gen_two):
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

def create_test_generator(df_a, df_b, img_shape, batch_size=1):
    test_gen1 = generator.flow_from_dataframe(
    df_a,
    batch_size=batch_size, 
    target_size=img_shape, 
    shuffle=False,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation")

    test_gen2 = generator.flow_from_dataframe(
        df_b,
        batch_size=1, 
        target_size=img_shape, 
        shuffle=False,
        color_mode="rgb",
        class_mode="categorical",
        subset="validation")
    
    test_steps = test_gen1.samples // batch_size

    multi_test_generator = create_data_generator(
        test_gen1,
        test_gen2,
    )

    # print and validate identical y_true labels
    [custom_print(test_gen1.filenames[i], i, x) for i, x in enumerate(test_gen1.classes[0:5])]

    [custom_print(test_gen2.filenames[i], i, x) for i, x in enumerate(test_gen2.classes[0:5])]

    y_true = test_gen1.classes

    

    return multi_test_generator, test_steps, y_true


def custom_print(f, i, x):
    print(f"index : {i}, y_true : {x} : filename : {f}")