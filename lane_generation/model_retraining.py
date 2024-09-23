import os
import numpy
import cv2
import matplotlib
import pandas
from PIL import Image
from tensorflow import keras
import tensorflow
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, LeakyReLU, BatchNormalization, Activation, Dropout, MaxPooling2D
from tensorflow.keras import layers, models
from tensorflow.python.client import device_lib
import matplotlib.pyplot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
import gc
import datetime
import threading
# width = 1640
width_standard= 1640
height_standard = 590
# width = 1152
# height = 192

width = 512
height = 128

fileXTrain = "./DataClean/Train/X"
fileYTrain = "./DataClean/Train/Y"
fileXVal = "./DataClean/Val/X"
fileYVal = "./DataClean/Val/Y"
fileXTest = "./DataTest/Data/X"
fileYTest = "./DataTest/Data/Y"

def read_data_file(path, quantity = False):
    images = []
    i = 0
    print(path)
    for file in os.listdir(path):
        file_name = os.path.join(path, file)
        # print(file_name)
        if (".jpg" in file):
            img = numpy.array(Image.open(file_name))
            images.append(img)
            del img
            # matplotlib.pyplot.imshow(img)
            # matplotlib.pyplot.show()
            i = i + 1
            if (quantity == i and quantity != False):
                break
    return images

train = [[],[]]
val = None
xImageTest = None
model = None
d = None
length_train = 10240

history_model_gen_file = None
def thread1():
    train[0] = numpy.array(read_data_file(fileXTrain, length_train))
def thread2():
    train[1] = numpy.array(read_data_file(fileYTrain, length_train))
def thread3():
    global val
    val = [numpy.array(read_data_file(fileXVal)), numpy.array(read_data_file(fileYVal))]
def thread4():
    global xImageTest
    xImageTest = read_data_file(fileXTest, 500)
    xImageTest = numpy.array(numpy.array(xImageTest).reshape(-1, height, width, 1))
def thread5():
    global model
    model = tensorflow.keras.models.load_model('model_gen_10k.h5')
    model.summary()
    model.compile(optimizer= Adam(lr=0.0002, beta_1=0.65),
                        loss='mean_squared_error',
                        metrics=['accuracy'])
def thread6():
    global d
    d = tensorflow.keras.models.load_model('discriminator_04-23_12-59_1.0.h5')
def thread7():
    global history_model_gen_file
    history_model_gen_file = pandas.read_csv("history_model_gen.csv")

thread1 = threading.Thread(target=thread1)
thread2 = threading.Thread(target=thread2)
thread3 = threading.Thread(target=thread3)
thread4 = threading.Thread(target=thread4)
thread5 = threading.Thread(target=thread5)
thread6 = threading.Thread(target=thread6)
thread7 = threading.Thread(target=thread7)

thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()
thread7.start()

thread1.join()
thread2.join()
thread3.join()
thread4.join()
thread5.join()
thread6.join()
thread7.join()
# train = [numpy.array((read_data_file(fileXTrain, 15000))), numpy.array(read_data_file(fileYTrain, 15000))]
# val = [numpy.array(read_data_file(fileXVal)), numpy.array(read_data_file(fileYVal))]

# xImageTest = read_data_file(fileXTest)
# xImageTest = numpy.array(numpy.array(xImageTest).reshape(-1, height, width, 1))

# model = tensorflow.keras.models.load_model('model_gen_17k_15-05_0.975.h5')
# # model.summary()
# model.compile(optimizer= Adam(lr=0.0002, beta_1=0.5),
#                             loss='mean_squared_error',
#                             metrics=['accuracy'])

# d = tensorflow.keras.models.load_model('discriminator_val-acc-0.998.h5')
# # d.summary()

# print(len(train[0]))
# print(len(train[1]))
# print(len(val[0]))
# print(len(val[1]))


def acc(d, n = 0.5):
    t = 0
    f = 0
    for i in d:
        if (i > n):
            t += 1
        else:
            f += 1
    return t / (t + f)

# history_model_gen_file = pandas.read_csv("history_model_gen.csv")

epochs = 1
batch_size=3

def train_model():
    history = model.fit(train[0], train[1], validation_data = (val[0], val[1]), epochs=epochs, batch_size=batch_size)
    time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    # model.save('model_gen_10k_' + time + '_' +str(round(history.history['val_accuracy'][-1], 3)) + '.h5')
    model.save('model_gen_10k.h5')
    # result = model.predict(xPImgTrain)s
    gc.collect()
    result = numpy.array(model.predict(xImageTest.reshape(-1, height, width, 1), batch_size=1))
    p = d.predict([xImageTest, result], batch_size=1)
    a = acc(p)
    del result
    del p
    print(a)
    return a, history
        
a = 0
while (a < 0.8):
    a, history = train_model()
    for i in range(epochs):
        history_model_gen_file.loc[len(history_model_gen_file)] = {
            "his_train_loss": history.history['loss'][i],
            "his_train_acc": history.history['accuracy'][i],
            "his_val_loss": history.history['val_loss'][i],
            "his_val_acc": history.history['val_accuracy'][i],
            "his_test_acc": a,
        }
    del history
    gc.collect()
    history_model_gen_file.to_csv("history_model_gen.csv", index=False)