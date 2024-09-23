import os
import numpy
import cv2
import matplotlib.pyplot
from PIL import Image
from tensorflow import keras
import tensorflow
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, LeakyReLU, BatchNormalization, Activation, Dropout, Flatten, Dense, MaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.python.client import device_lib
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
import random
# from sklearn.model_selection import train_test_split
import sklearn.model_selection
import gc
import datetime
import pandas
import threading

# width = 1640
width_standard= 1640
height_standard = 590
# width = 1152
# height = 192

width = 512
height = 128

fileTrain = "./DataTrain"
fileVal = "./DataVal"
fileSaveTrain = "./DataClean/Train"
fileSaveVal = "./DataClean/Val"
fileSave = "./DataClean"

fileXTrain = "./DataClean/Train/X"
fileYTrain = "./DataClean/Train/Y"
fileXVal = "./DataClean/Val/X"
fileYVal = "./DataClean/Val/Y"

fileDiscriminatorRaw = "./DataDiscriminator/Raw"
fileDataDiscriminatorTrue = "./DataDiscriminator/DataTrue"

fileDataDiscriminatorXTrue = "./DataDiscriminator/DataTrue/X"
fileDataDiscriminatorYTrue = "./DataDiscriminator/DataTrue/Y"
fileDataDiscriminatorPredict = "./DataDiscriminator/DataFalse"

def read_data_file(path, quantity = False, standardize = False):
    images = []
    print(path)
    i = 0
    for file in os.listdir(path):
        file_name = os.path.join(path, file)
        # print(file_name)
        if (".jpg" in file):
            img = numpy.array(Image.open(file_name))
            if (quantity != False and i == quantity):
                break
            i += 1
            if (standardize):
                img_stand = img / 255
                images.append(img_stand)
                del img_stand
                del img
                # gc.collect()
                continue
            images.append(img)
            del img
            # matplotlib.pyplot.imshow(img)
            # matplotlib.pyplot.show()
    return images

TestX = None
TestY = None

def thread1():
    global TestX
    TestX = read_data_file(fileDataDiscriminatorXTrue, 4096)
    TestX.extend(TestX)

def thread2():
    global TestY
    TestY = read_data_file(fileDataDiscriminatorYTrue, 4096) + read_data_file(fileDataDiscriminatorPredict, 4096, True)

thread1 = threading.Thread(target=thread1)
thread2 = threading.Thread(target=thread2)

thread1.start()
thread2.start()

thread1.join()
thread2.join()

print(TestY[0])
print(TestY[len(TestY) - 1])
# TestY = read_data_file(fileDataDiscriminatorYTrue)
# TestY = read_data_file(fileDataDiscriminatorPredict, True)
gc.collect()
# TestX.extend(TestX)

if (len(TestX) != len(TestY)):
    print(len(TestX))
    print(len(TestY))
    print("Error")

label = []
length = int(len(TestY) / 2)

for i in range(length):
    label.append(0)
for i in range(length):
    label.append(1)


discriminatorData = [TestX, TestY, label]

discriminatorData = [list(row) for row in zip(*discriminatorData)]
random.shuffle(discriminatorData)
random.shuffle(discriminatorData)
random.shuffle(discriminatorData)
random.shuffle(discriminatorData)

discriminatorTrain, discriminatorVal = sklearn.model_selection.train_test_split(discriminatorData, test_size=0.12)

discriminatorTrain = [list(row) for row in zip(*discriminatorTrain)]
discriminatorVal = [list(row) for row in zip(*discriminatorVal)]

print(len(discriminatorTrain[0]))
print(len(discriminatorVal[0]))

def _discriminator():
    init = RandomNormal(stddev=0.02)

    input_image = Input(shape=(height, width, 1))
    input_target = Input(shape=(height, width, 1))
    merge = Concatenate()([input_image, input_target])

    d = Conv2D(8, (3,3), padding='same', activation=LeakyReLU(alpha=0.2), kernel_initializer=init)(merge) 
    d = Dropout(0.1)(d)
    d = MaxPooling2D((2, 2))(d)
    d = BatchNormalization()(d, training=True)

    d = Conv2D(16, (3,3), padding='same', activation=LeakyReLU(alpha=0.2), kernel_initializer=init)(d)
    d = Dropout(0.2)(d)
    d = MaxPooling2D((2, 2))(d)
    d = BatchNormalization()(d, training=True)

    d = Conv2D(32, (3,3), padding='same', activation=LeakyReLU(alpha=0.2), kernel_initializer=init)(d)
    d = Dropout(0.2)(d)
    d = MaxPooling2D((2, 2))(d)
    d = BatchNormalization()(d, training=True)
    
    d = Conv2D(64, (3,3), padding='same', activation=LeakyReLU(alpha=0.2), kernel_initializer=init)(d)
    d = Dropout(0.25)(d)
    d = MaxPooling2D((2, 2))(d)
    d = BatchNormalization()(d, training=True)
    
    d = Conv2D(128, (3,3), padding='same', activation=LeakyReLU(alpha=0.2), kernel_initializer=init)(d)
    d = Dropout(0.25)(d)
    d = MaxPooling2D((2, 2))(d)
    d = BatchNormalization()(d, training=True)
    
    d = Conv2D(256, (3,3), padding='same', activation=LeakyReLU(alpha=0.2), kernel_initializer=init)(d)
    d = Dropout(0.3)(d)
    d = MaxPooling2D((2, 2))(d)

    d = Conv2D(512, (3,3), padding='same', activation=LeakyReLU(alpha=0.2), kernel_initializer=init)(d)
    d = MaxPooling2D((2, 2))(d)
    # d = Conv2D(1, (3,3), padding='same', activation='sigmoid', kernel_initializer=init)(d)
    d = Flatten()(d)
    d = Dense(2048, activation='relu')(d)
    d = Dense(512, activation='relu')(d)
    d = Dense(128, activation='relu')(d)
    d = Dense(32, activation='relu')(d)
    d = Dense(8, activation='relu')(d)
    d = Dense(1, activation='sigmoid')(d)

    model = Model([input_image, input_target], d)

    return model

discriminator = _discriminator()
discriminator.summary()

history_model_gen_file = pandas.read_csv("history_model_discriminator.csv")

epochs = 10
batch_size = 16

def TySo(data):
    f = 0
    l = len(data)
    for i in range(l):
        if (data[i] == 0):
            f += 1
    return f / l

print(TySo(discriminatorTrain[2]))
print(TySo(discriminatorVal[2]))

discriminator.compile(optimizer = Adam(lr=0.0002, beta_1=0.7), loss='binary_crossentropy', metrics=['accuracy'])
while (True):
    history = discriminator.fit(x=[numpy.array(discriminatorTrain[0]), numpy.array(discriminatorTrain[1])], 
                    y=numpy.array(discriminatorTrain[2]), 
                    validation_data=([numpy.array(discriminatorVal[0]), numpy.array(discriminatorVal[1])], numpy.array(discriminatorVal[2])),
                    epochs=epochs, 
                    batch_size=batch_size)
    time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    discriminator.save('discriminator_' + time + '_' + str(round(history.history['val_accuracy'][-1], 3)) + '.h5')
    for i in range(epochs):
        history_model_gen_file.loc[len(history_model_gen_file)] = {
            "his_train_loss": history.history['loss'][i],
            "his_train_acc": history.history['accuracy'][i],
            "his_val_loss": history.history['val_loss'][i],
            "his_val_acc": history.history['val_accuracy'][i],
        }
    history_model_gen_file.to_csv("history_model_discriminator.csv", index=False)

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']
# epochs = range(1, len(loss) + 1)
# # print(loss)
# # print(val_loss)
# # print(accuracy)
# # print(val_accuracy)
# # print(epochs)

# final_accuracy = history.history['val_accuracy'][-1]

# matplotlib.pyplot.figure(figsize=(10, 5))
# matplotlib.pyplot.plot(epochs, loss, 'b', label='Training loss')
# matplotlib.pyplot.plot(epochs, val_loss, 'r', label='Validation loss')
# matplotlib.pyplot.title('Training and validation loss')
# matplotlib.pyplot.xlabel('Epochs')
# matplotlib.pyplot.ylabel('Loss')
# matplotlib.pyplot.legend()
# matplotlib.pyplot.grid(True)

# matplotlib.pyplot.show()

# matplotlib.pyplot.figure(figsize=(10, 5))
# matplotlib.pyplot.plot(epochs, accuracy, 'b', label='Training accuracy')
# matplotlib.pyplot.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
# matplotlib.pyplot.title('Training and validation accuracy')
# matplotlib.pyplot.xlabel('Epochs')
# matplotlib.pyplot.ylabel('Accuracy')
# matplotlib.pyplot.legend()
# matplotlib.pyplot.grid(True)

# matplotlib.pyplot.show()

# # Vẽ biểu đồ accuracy
# matplotlib.pyplot.figure(figsize=(10, 5))
# matplotlib.pyplot.plot(epochs, accuracy, 'b', label='Training accuracy')
# matplotlib.pyplot.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
# matplotlib.pyplot.title('Training and validation accuracy')
# matplotlib.pyplot.xlabel('Epochs')
# matplotlib.pyplot.ylabel('Accuracy')
# matplotlib.pyplot.legend()
# matplotlib.pyplot.grid(True)
# matplotlib.pyplot.show()