# pylint:disable=no-member (Removes linting problems with cv)

# Installing `caer` and `canaro` since they don't come pre-installed
# Uncomment the following line:
# !pip install --upgrade caer canaro

import gc
import os
import time

import caer
import canaro
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical

# Best size of images
IMG_SIZE = (80, 80)
# Since we don't require color in our images, set this to 1, grayscale
channels = 1
char_path = r"../input/the-simpsons-characters-dataset/simpsons_dataset"

# Creating a character dictionary, sorting it in descending order
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

# Sort in descending order
char_dict = caer.sort_dict(char_dict, descending=True)
char_dict

#  Getting the first 10 categories with the most number of images
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
characters

# Create the training data
train = caer.preprocess_from_dir(
    char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True
)

# Number of training samples
len(train)

# Visualizing the data (OpenCV doesn't display well in Jupyter notebooks)
plt.figure(figsize=(30, 30))
plt.imshow(train[0][0], cmap="gray")
plt.show()

# Separating the array and corresponding labels
# Reshape the feature set to a 4d tensor so that it can be fed into the model with no restrictions whatsoever
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)


# Normalize the featureSet ==> (0,1)
featureSet = caer.normalize(featureSet)
# Converting numerical labels to binary class vectors
labels = to_categorical(labels, len(characters))


# Creating train and validation data
# val_ratio=0.2: 20% of the data will go to validation set, 80% goes to training set
x_train, x_val, y_train, y_val = caer.train_test_split(
    featureSet, labels, val_ratio=0.2
)

# Deleting variables to save memory
del train
del featureSet
del labels
gc.collect()

# https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
# Useful variables when training
# Definition: The batch size is a number of samples processed before the model is updated.
BATCH_SIZE = 32
# An epoch is a term used in machine learning and indicates the number of passes of the entire training dataset the machine learning algorithm has completed. Datasets are usually grouped into batches (especially when the amount of data is very large).
# Definition: The number of epochs is the number of complete passes through the training dataset.
EPOCHS = 10

# Image data generator (introduces randomness in network ==> better accuracy)
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Create our model (returns the compiled model)
model = canaro.models.createSimpsonsModel(
    IMG_SIZE=IMG_SIZE,
    channels=channels,
    output_dim=len(characters),
    loss="binary_crossentropy",
    decay=1e-7,
    learning_rate=0.001,
    momentum=0.9,
    nesterov=True,
)

model.summary()

# Training the model

print("[INFO] training face recognizer...")
start = time.time()
# Schedule the learning rate at specific intervals so that our network will train better
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]
training = model.fit(
    train_gen,
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    validation_steps=len(y_val) // BATCH_SIZE,
    callbacks=callbacks_list,
)
end = time.time()
print("[INFO] training took {:.4f} seconds".format(end - start))

print(characters)


"""## Testing"""

test_path = r"../input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_0.jpg"

img = cv.imread(test_path)

plt.imshow(img)
plt.show()


def prepare(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, IMG_SIZE)
    image = caer.reshape(image, IMG_SIZE, 1)
    return image


predictions = model.predict(prepare(img))

# Getting class with the highest probability
print(characters[np.argmax(predictions[0])])
