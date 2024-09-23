import os
import numpy
import cv2
import matplotlib
import pandas
import matplotlib.pyplot
from PIL import Image
from tensorflow import keras
import tensorflow
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, LeakyReLU, BatchNormalization, Activation, Dropout, MaxPooling2D
from tensorflow.keras import models, layers
from tensorflow.python.client import device_lib
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

def endcoder_block(layer_in, n_filters, batchnorm = True):
    init = RandomNormal(stddev=0.02)
    e = Conv2D(n_filters, (4, 4), strides=(1,1), padding='same', activation=LeakyReLU(alpha=0.2), kernel_initializer=init)(layer_in)
    e = MaxPooling2D((2, 2))(e)
    if batchnorm:
        e = BatchNormalization()(e, training=True)
    return e


def decode_block(layer_in, skip_in, n_filters, dropout = True, concatenate = True):
    init = RandomNormal(stddev=0.02)
    d = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', activation=LeakyReLU(alpha=0.2), kernel_initializer=init)(layer_in)
    d = BatchNormalization()(d, training=True)
    if dropout:
      d = Dropout(0.15)(d, training=True)
    if concatenate:
        d = Concatenate()([d, skip_in])
    return d

def _gen():
    init = RandomNormal(stddev=0.02)
    
    input_image = Input(shape=(height, width, 1))
    e1 = endcoder_block(input_image, 64, batchnorm=False)
    e2 = endcoder_block(e1, 128)
    e3 = endcoder_block(e2, 256)
    e4 = endcoder_block(e3, 512)
    e5 = endcoder_block(e4, 512)
    e6 = endcoder_block(e5, 1024)

    b = Conv2D(2048, (4, 4), strides=(2,2), padding='same', activation='relu', kernel_initializer=init)(e6)

    d1 = decode_block(b, e6, 1024)
    d2 = decode_block(d1, e5, 512)
    d3 = decode_block(d2, e4, 512)
    d4 = decode_block(d3, e3, 256, dropout=False)
    d5 = decode_block(d4, e2, 128, dropout=False, concatenate=False)
    d6 = decode_block(d5, e1, 64, dropout=False, concatenate=False)

    out_image = Conv2DTranspose(1, (1, 1), strides=(2,2), activation='tanh', kernel_initializer=init)(d6)

    model = Model(input_image, out_image)

    return model

train = [[],[]]
val = None
xImageTest = None
model = None
d = None
history_model_gen_file = None

def thread1():
    train[0] = numpy.array(read_data_file(fileXTrain, 10240))
def thread2():
    train[1] = numpy.array(read_data_file(fileYTrain, 10240))
def thread3():
    global val
    val = [numpy.array(read_data_file(fileXVal)), numpy.array(read_data_file(fileYVal))]
def thread4():
    global xImageTest
    xImageTest = read_data_file(fileXTest, 500)
    xImageTest = numpy.array(numpy.array(xImageTest).reshape(-1, height, width, 1))
def thread5():
    global model
    model = _gen()
    model.compile(optimizer= Adam(lr=0.0002, beta_1=0.8),
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

# train = [numpy.array((read_data_file(fileXTrain))), numpy.array(read_data_file(fileYTrain))]
# val = [numpy.array(read_data_file(fileXVal)), numpy.array(read_data_file(fileYVal))]

# xImageTest = read_data_file(fileXTest, 500)
# xImageTest = numpy.array(numpy.array(xImageTest).reshape(-1, height, width, 1))

# model = _gen()
# model.summary()
# model.compile(optimizer= Adam(lr=0.0002, beta_1=0.5),
#                             loss='mean_squared_error',
#                             metrics=['accuracy'])

# d = tensorflow.keras.models.load_model('discriminator_val-acc-0.998.h5')
# d.summary()

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

# columns = ["his_train_loss", "his_train_acc", "his_val_loss", "his_val_acc", "his_test_acc"]
# history_model_gen_file = pandas.DataFrame(columns=columns)

epochs = 1
batch_size=3

def train_model():
    history = model.fit(train[0], train[1], validation_data = (val[0], val[1]), epochs=epochs, batch_size=batch_size)
    time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    model.save('model_gen_10k_' + time + '_' +str(round(history.history['val_accuracy'][-1], 3)) + '.h5')
    # result = model.predict(xPImgTrain)
    result = numpy.array(model.predict(xImageTest.reshape(-1, height, width, 1), batch_size=2))
    p = d.predict([xImageTest, result], batch_size=2)
    a = acc(p)
    for i in range(epochs):
        data_histroy_model = {
            "his_train_loss": history.history['loss'][i],
            "his_train_acc": history.history['accuracy'][i],
            "his_val_loss": history.history['val_loss'][i],
            "his_val_acc": history.history['val_accuracy'][i],
            "his_test_acc": a,
        }
        history_model_gen_file.loc[len(history_model_gen_file)] = data_histroy_model
    history_model_gen_file.to_csv("history_model_gen.csv", index=False)
    del result
    del history
    del p
    print(a)
    return a
        
a = 0
while (a < 0.8):
    a = train_model()
    gc.collect()