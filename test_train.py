import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers

width = 40
height = 15
TRAIN_DATA = 'data/lane/train'

Xtrain = []
Ytrain = []
dict = {'right': [1, 0], 'wrong': [0, 1]}

def rotate_image(image, label):
    list_image = []
    list_label = []
    count = 0
    center = (width // 2, height // 2)
    for distance in range(-8, 10, 2):
        translated_image = cv2.warpAffine(image, np.float32([[1, 0, distance], [0, 1, 0]]), (width, height))
        for degrees in range(-15, 20, 5):
            distance_rotated_image = cv2.warpAffine(
                translated_image, cv2.getRotationMatrix2D(center, degrees, 1.0), (width, height)
            )
            list_image.append(distance_rotated_image)
            list_label.append(label)
            count = count + 1
    return list_image, list_label

def DocDuLieu(file):
    DuLieu = []
    Label = []
    label = ''
    for filename in os.listdir(file):
        filename_path = os.path.join(file, filename)
        list_filename_sub_path = []
        label = filename
        for filename_sub in os.listdir(filename_path):
            if (".jpg" in filename_sub or ".png" in filename_sub):
                filename_sub_path = os.path.join(filename_path, filename_sub)
                img = np.array(Image.open(filename_sub_path))
                img = cv2.resize(img, (width, height))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                list_rotate_image, list_label = rotate_image(img, dict[label])
                list_filename_sub_path.extend(list_rotate_image)
                Label.extend(list_label)
        DuLieu.extend(list_filename_sub_path)
    return DuLieu, Label

Xtrain, Ytrain = DocDuLieu(TRAIN_DATA)

print(len(Xtrain))
print(len(Ytrain))


model = tf.keras.models.Sequential([
    layers.Conv2D(3, (3, 3), input_shape=(height, width, 1), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.15),
    layers.Flatten(),
    layers.Dense(20, activation='relu'),
    layers.Dense(2, activation='softmax'),
])

model.summary()

early_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", min_delta=0, patience=10, verbose=1, mode="auto"
)

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Limit GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # Set the desired memory limit (in MB)
    except RuntimeError as e:
        print(e)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    distributed_model = model

distributed_model.fit(
    np.array(Xtrain), np.array(Ytrain), epochs=150, batch_size=1000,
    callbacks=[early_callback],
    verbose=True
)

model.save('model_demo_1_10epochs.h5')
    
