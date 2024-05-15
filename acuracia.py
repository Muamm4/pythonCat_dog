from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Other libraries
import os
import random

val_ds = keras.utils.image_dataset_from_directory(
    directory = './training_set/training_set',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
)
train_ds = keras.utils.image_dataset_from_directory(
    directory = './test_set/test_set',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
)

def create_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Carregar o modelo previamente salvo
model = create_model()
checkpoint_path = "checkpoint/cp.weights.h5"
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
model.summary()
test_loss, test_acc = model.evaluate(train_ds, verbose=1)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
