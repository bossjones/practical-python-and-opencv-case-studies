# pylint:disable=no-member
from collections import Counter
import glob
import os
import pathlib
import time

from typing import Any, List

import caer
import canaro
import cv2
import h5py
import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
# from keras.optimizers import Adam
from keras.optimizer_v2.gradient_descent import SGD
import keras.optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from rich import inspect as rich_inspect, print as rich_print
from sklearn.model_selection import train_test_split
import tensorflow as tf

from practical_python_and_opencv_case_studies.dataset_builder import constants, utils

ROOT_DIR = os.path.dirname(__file__)

# # constants.pic_size = 64
# constants.pic_size = 80
# batch_size = 32
# # epochs = 200
# epochs = 200
# num_classes = len(constants.map_characters)
# constants.pictures_per_class = 1000
# constants.test_size = 0.15



def load_pictures(BGR):
    """
    Load pictures from folders for characters from the constants.map_characters dict and create a numpy dataset and
    a numpy labels set. Pictures are re-sized into picture_size square.
    :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
    :return: dataset, labels set
    """
    pics = []
    labels = []
    for k, char in constants.map_characters.items():
        pictures = [k for k in glob.glob(f"{constants.characters_folder}/%s/*" % char)]
        pictures = utils.filter_images(pictures)
        nb_pic = (
            round(constants.pictures_per_class / (1 - constants.test_size))
            if round(constants.pictures_per_class / (1 - constants.test_size)) < len(pictures)
            else len(pictures)
        )
        # nb_pic = len(pictures)
        for pic in np.random.choice(pictures, nb_pic):
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = cv2.resize(a, (constants.pic_size, constants.pic_size))
            pics.append(a)
            labels.append(k)
    return np.array(pics), np.array(labels)


def get_dataset(save=False, load=False, BGR=False):
    """
    Create the actual dataset split into train and test, pictures content is as float32 and
    normalized (/255.). The dataset could be saved or loaded from h5 files.
    :param save: saving or not the created dataset
    :param load: loading or not the dataset
    :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
    :return: X_train, X_test, y_train, y_test (numpy arrays)
    """
    if load:
        h5f = h5py.File(f"{ROOT_DIR}/dataset.h5", "r")
        X_train = h5f["X_train"][:]
        X_test = h5f["X_test"][:]
        h5f.close()

        h5f = h5py.File(f"{ROOT_DIR}/labels.h5", "r")
        y_train = h5f["y_train"][:]
        y_test = h5f["y_test"][:]
        h5f.close()
    else:
        X, y = load_pictures(BGR)
        y = tf.keras.utils.to_categorical(y, constants.num_classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=constants.test_size)
        if save:
            h5f = h5py.File(f"{ROOT_DIR}/dataset.h5", "w")
            h5f.create_dataset("X_train", data=X_train)
            h5f.create_dataset("X_test", data=X_test)
            h5f.close()

            h5f = h5py.File(f"{ROOT_DIR}/labels.h5", "w")
            h5f.create_dataset("y_train", data=y_train)
            h5f.create_dataset("y_test", data=y_test)
            h5f.close()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)
    if not load:
        dist = {
            k: tuple(
                d[k]
                for d in [
                    dict(Counter(np.where(y_train == 1)[1])),
                    dict(Counter(np.where(y_test == 1)[1])),
                ]
            )
            for k in range(constants.num_classes)
        }
        print(
            "\n".join(
                [
                    "%s : %d train pictures & %d test pictures"
                    % (constants.map_characters[k], v[0], v[1])
                    for k, v in sorted(
                        dist.items(), key=lambda x: x[1][0], reverse=True
                    )
                ]
            )
        )
    return X_train, X_test, y_train, y_test


def create_model_four_conv(input_shape):
    """
    CNN Keras model with 4 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(constants.num_classes))
    model.add(Activation("softmax"))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt


def create_model_six_conv(input_shape):
    """
    CNN Keras model with 6 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(constants.num_classes, activation="softmax"))
    opt = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, opt


def load_model_from_checkpoint(
    weights_path, six_conv=False, input_shape=(constants.pic_size, constants.pic_size, 3)
):
    if six_conv:
        model, opt = create_model_six_conv(input_shape)
    else:
        model, opt = create_model_four_conv(input_shape)
    model.load_weights(weights_path)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def lr_schedule(epoch):
    lr = 0.01
    return lr * (0.1 ** int(epoch / 10))


def training(model, X_train, X_test, y_train, y_test, data_augmentation=True):
    """
    Training.
    :param model: Keras sequential model
    :param data_augmentation: boolean for data_augmentation (default:True)
    :param callback: boolean for saving model checkpoints and get the best saved model
    :param six_conv: boolean for using the 6 convs model (default:False, so 4 convs)
    :return: model and epochs history (acc, loss, val_acc, val_loss for every epoch)
    """
    if data_augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,
        )  # randomly flip images
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
        # NOTE: Todo, try this instead maybe?
        # SOURCE: https://github.com/ceo1207/SimpsonRecognition/commit/db854078d9964f640cb95f9b3a3023707a03ca31
        # filepath = "checkpoint-{epoch:02d}-{val_acc:.2f}.hdf5"
        filepath = f"{ROOT_DIR}/weights_6conv_%s.hdf5" % time.strftime("%d%m/%Y")
        checkpoint = ModelCheckpoint(
            filepath, monitor="val_accuracy", verbose=0, save_best_only=True, mode="max"
        )
        callbacks_list = [LearningRateScheduler(lr_schedule), checkpoint]
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=constants.batch_size),
            steps_per_epoch=X_train.shape[0] // constants.batch_size,
            epochs=40,
            validation_data=(X_test, y_test),
            callbacks=callbacks_list,
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            batch_size=constants.batch_size,
            epochs=constants.epochs,
            validation_data=(X_test, y_test),
            shuffle=True,
        )
    return model, history


def get_character_dict_and_top_10():
    # Creating a character dictionary, sorting it in descending order
    char_dict = {}
    for char in os.listdir(f"{constants.characters_folder}"):
        if ".DS_Store" in char:
            continue
        char_dict[char] = len(
            os.listdir(os.path.join(f"{constants.characters_folder}", char))
        )

    # Sort in descending order
    char_dict = caer.sort_dict(char_dict, descending=True)
    rich_print(f"[dumping] char_dict")
    rich_print(char_dict)

    #  Getting the first 10 categories with the most number of images
    characters = []
    count = 0
    for i in char_dict:
        characters.append(i[0])
        count += 1
        if count >= 10:
            break
    return char_dict, characters


def create_training_data(
    char_path="",
    characters=[],
    channels=constants.channels,
    IMG_SIZE=constants.IMG_SIZE,
):
    train = caer.preprocess_from_dir(
        char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True
    )

    rich_print("Number of training samples")
    rich_print(f"{len(train)}")

    return train


def var_print(msg: str, var: Any):
    rich_print(f"{msg}")
    rich_print(f"{var}")


if __name__ == "__main__":
    # Catch exception
    import sys

    from IPython.core import ultratb
    from IPython.core.debugger import Tracer  # noqa
    # from bpython.bpdb.debugger import BPdb
    import bpdb
    import bpython

    # sys.excepthook = ultratb.FormattedTB(
    #     mode="Verbose", color_scheme="Linux", call_pdb=True, ostream=sys.__stdout__, debugger_cls=bpdb.BPdb
    # )
    sys.excepthook = ultratb.FormattedTB(
        mode="Verbose", color_scheme="Linux", call_pdb=True, ostream=sys.__stdout__
    )

    # from practical_python_and_opencv_case_studies.dataset_builder import train

    # get data from the directory containing characters images
    # first time  use load=False, save=True
    X_train, X_test, y_train, y_test = get_dataset(save=True, load=False)
    # X_train, X_test, y_train, y_test = get_dataset(load=True)
    # second time  use load=True, save=false
    # X_train, X_test, y_train, y_test = get_dataset(save=False, load=True)
    model, opt = create_model_six_conv(X_train.shape[1:])
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model, history = training(
        model, X_train, X_test, y_train, y_test, data_augmentation=True
    )

    # # -----------------------------------------------------------
    # # 2021 stuff
    # # -----------------------------------------------------------
    # char_dict, characters = get_character_dict_and_top_10()

    # training_data = create_training_data(
    #     char_path=f"{constants.characters_folder}", characters=characters
    # )

    # var_print("dump training data", training_data)

    # # -----------------------------------------------------------
