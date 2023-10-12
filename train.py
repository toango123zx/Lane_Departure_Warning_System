import numpy
import os
import cv2
import time
import matplotlib.pyplot
from PIL import Image
from tensorflow import keras
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models

from sklearn.model_selection import train_test_split
width = 160
height = 60
TRAIN_DATA = '../data/lane/train'
VAL_DATA = '../data/lane/acc'

Xtrain =[]
Ytrain = []
dict = {'right': [1, 0], 'wrong': [0, 1]}

def rotate_image(image, label):
    list_image = []
    list_label = []
    count = 0
    center = (width // 2, height // 2)
    for distance in range (-8, 10, 2):
        translated_image = cv2.warpAffine(image, numpy.float32([[1, 0, distance], [0, 1, 0]]) , (width, height))
        for degrees in range(-15, 20, 5):
            distance_rotated_image = cv2.warpAffine(translated_image, cv2.getRotationMatrix2D(center, degrees, 1.0), (width, height))
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
                img = numpy.array(Image.open(filename_sub_path))
                img = cv2.resize(img, (width, height))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                list_rotate_image, list_label = rotate_image(img, dict[(label)])
                list_filename_sub_path.extend(list_rotate_image)
                Label.extend(list_label)
        DuLieu.extend(list_filename_sub_path)
    return DuLieu, Label

Xtrain, Ytrain = DocDuLieu(TRAIN_DATA)
# Xtrain = numpy.array(Xtrain)
# Xtrain = Xtrain.astype(numpy.float32) / 255.0
Xval, Yval = DocDuLieu(VAL_DATA)
# Xval = numpy.array(Xval)
# Xval = Xval.astype(numpy.float32) / 255.0

print(len(Xtrain))
print(len(Ytrain))


model_training_frist = models.Sequential([
    layers.Conv2D(4, (3, 3), input_shape=(height, width, 1), activation = 'relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.15),
    
    layers.Conv2D(8, (3, 3), activation = 'relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(16, (3, 3), activation = 'relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(2000, activation = 'relu'),
    layers.Dense(1000, activation = 'relu'),
    layers.Dense(500, activation = 'relu'),
    layers.Dense(100, activation = 'relu'),
    layers.Dense(2, activation = 'softmax'),
])

model_training_frist.summary()
model_training_frist.compile(optimizer='SGD',
                             loss='mse',
                             metrics=['accuracy'])



early_callback = tensorflow.keras.callbacks.EarlyStopping(monitor="loss", min_delta= 0 , patience=10, verbose=1, mode="auto")

# Xtrain = Xtrain.astype(np.float32) / 255.0

history = model_training_frist.fit(numpy.array(Xtrain), numpy.array(Ytrain), validation_data = (numpy.array(Xval), numpy.array(Yval)), epochs=300, batch_size=1200,
                         callbacks = [early_callback],
                         verbose=True)

model_training_frist.save('model_demo_1_10epochs.h5')


# list all data in history
print(history.history.keys())
# summarize history for loss mae
matplotlib.pyplot.plot(history.history['loss'], color='red')
# matplotlib.pyplot.plot(history.history)
# matplotlib.pyplot.title('model m')
matplotlib.pyplot.ylabel('loss')
matplotlib.pyplot.xlabel('epoch')
matplotlib.pyplot.show()
# summarize history for loss mse
matplotlib.pyplot.plot(history.history['accuracy'], color='blue')
matplotlib.pyplot.ylabel('accuracy')
matplotlib.pyplot.xlabel('epoch')
matplotlib.pyplot.show()