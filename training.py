# For data manipulation
import numpy as np 
import pandas as pd 

# For data visualization
import matplotlib.pyplot as plt 
import seaborn as sns

# Ingore the warnings
import warnings
warnings.filterwarnings('ignore')

# DL Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import image_dataset_from_directory

# Other libraries
import os
import random

# Load the image dataset from the directory using utils

# dados para treino
train_ds = keras.utils.image_dataset_from_directory(
    directory = './test_set/test_set',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
)

# dados para validação
val_ds = keras.utils.image_dataset_from_directory(
    directory = './training_set/training_set',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
)

def visualize_images(path, num_images=5):
    # Get a list of image filenames
    image_filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if not image_filenames:
        raise ValueError("No images found in the specified path")

    # Select random images
    selected_images = random.sample(image_filenames, min(num_images, len(image_filenames)))

    # Create a figure and axes
    fig, axes = plt.subplots(1, num_images, figsize=(20, 4), facecolor='white')

    # Display each image
    for i, image_filename in enumerate(selected_images):
        # Load image
        image_path = os.path.join(path, image_filename)
        image = plt.imread(image_path)

        # Display image
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(image_filename)  # Set image filename as title

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Nome da Classes
classes = train_ds.class_names

# Iterating through each class to plot its images
for label in classes:
    # Specify the path containing the images to visualize
    path_to_visualize = f"./training_set/training_set/{label}"
    # Visualize 5 random images
    print(label.upper())
    visualize_images(path_to_visualize, num_images=5)

## modelo para tratamento de imagem
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

# Salvar os Dados
checkpoint_path = "checkpoint/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
model = create_model()
# Load the latest checkpoint
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
else:
    os.mkdir(checkpoint_dir)

model.summary()
# Train the model
history = model.fit(train_ds, batch_size=32, epochs=10, validation_data=val_ds, verbose=1, callbacks=[cp_callback])
test_loss, test_acc = model.evaluate(train_ds, verbose=2)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")



