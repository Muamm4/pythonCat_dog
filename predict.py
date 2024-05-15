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

# test_loss, test_acc = model.evaluate(train_ds, verbose=2)
# print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
# Função para carregar e preparar a imagem
def load_and_prepare_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)  # Convertendo a imagem para um array
    img_array = np.expand_dims(img_array, axis=0)  # Adicionando uma dimensão para o batch
    return img_array

# Caminho para a imagem a ser predita
image_path = './predict/Cat Breeds.jpeg'

# Fazer a predição
predictions = model.predict(load_and_prepare_image(image_path))

# Certificar-se de extrair o valor escalar antes de fazer mais operações
predicted_prob = predictions[0][0]  # Extrair o primeiro elemento do primeiro batch
print(f'Probabilidade prevista: {predicted_prob}')
# Se o modelo for classificação binária (sigmoid na camada de saída)
predicted_class = int(predicted_prob > 0.5)  # Convertendo probabilidades em classe binária
# Para obter o nome da classe, se você tiver uma lista de nomes de classes:
class_names = ['Cats', 'Dogs']

textstr = f'Classe predita: {class_names[predicted_class]}'
# Mostrar a imagem
img = np.asarray(Image.open(image_path))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95,s=textstr, fontsize=14,horizontalalignment='left', verticalalignment='bottom', bbox=props)
plt.imshow(img)
plt.show()
