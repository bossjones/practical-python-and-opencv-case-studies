# pylint:disable=no-member
import os
import pathlib

from typing import List

# from keras.callbacks import LearningRateScheduler, ModelCheckpoint
# from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
# from keras.models import Sequential
# # from keras.optimizers import Adam
# from keras.optimizer_v2.gradient_descent import SGD
# import keras.optimizers
# from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# import numpy as np
# from rich import inspect as rich_inspect, print as rich_print
# from sklearn.model_selection import train_test_split
# import tensorflow as tf


ROOT_DIR = os.path.dirname(__file__)


JSON_EXTENSIONS = [".json", ".JSON"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".MP4", ".MOV"]
AUDIO_EXTENSIONS = [".mp3", ".MP3"]
GIF_EXTENSIONS = [".gif", ".GIF"]
MKV_EXTENSIONS = [".mkv", ".MKV"]
M3U8_EXTENSIONS = [".m3u8", ".M3U8"]
WEBM_EXTENSIONS = [".webm", ".WEBM"]
# IMAGE_EXTENSIONS = [".png", ".jpeg", ".jpg", ".gif", ".PNG", ".JPEG", ".JPG", ".GIF"]
IMAGE_EXTENSIONS = [".png", ".jpeg", ".jpg", ".PNG", ".JPEG", ".JPG", ".GIF"]


def filter_images(file_system: List[str]) -> List[str]:
    file_system_images_only = []
    for f in file_system:
        p = pathlib.Path(f"{f}")
        if p.suffix in IMAGE_EXTENSIONS:
            file_system_images_only.append(f)
    return file_system_images_only
