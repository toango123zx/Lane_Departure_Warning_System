import os
import numpy
import cv2
import matplotlib
from PIL import Image
from tensorflow import keras
import tensorflow
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, LeakyReLU, BatchNormalization, Activation, Dropout
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.python.client import device_lib
import matplotlib

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

def regon_of_interest(image):
    polygons = numpy.array([[(0, height_standard), (0, 300), (700, 240), (850, 240), (width_standard, 300), (width_standard, height_standard)]])
    mask = numpy.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


def crop_image(image):
    return numpy.array(Image.fromarray(image).crop((0, 240, width_standard, height_standard)))

x = 0
y = 0

def read_data_file(path, fileSave, quantity = None):
    global x, y
    for file in os.listdir(path):
        if (bool(quantity) and x == quantity and y == quantity): 
            break
        file_name = os.path.join(path, file)
        # print(file_name)
        if (".jpg" in file):
            if (not (x == y or x + 1 == y)): 
                break
            file_save_train_name = fileSave + "/X/" + str(x) + ".jpg"
            img = numpy.array(Image.open(file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = regon_of_interest(img)
            img = crop_image(img)
            img = cv2.GaussianBlur(src=img, ksize=(5, 5), sigmaX=0, sigmaY=0)
            img = cv2.resize(img, (width, height))
            img = cv2.Canny(img, 50, 150)
            img = img / 255.0
            cv2.imwrite(file_save_train_name, img)
            x = x + 1
            # images_train.append(img)
            # matplotlib.pyplot.imshow(img)
            # matplotlib.pyplot.show()
        if (".txt" in file):
            with open(file_name, 'r') as f:
                if (not (x == y or x == y + 1)): 
                    break
                file_save_name = fileSave + "/Y/" + str(y) + ".jpg"
                label_x =[]
                labels = f.read()
                labels = labels.split('\n')
                if (labels[-1] == ''): labels.pop()
                labels = [[word] for word in labels]
                length = len(labels)
                # print(length)
                for i in range(0, length, 1):
                    numbers = [float(num) for num in labels[i][0].split()]
                    sub_arrays = [numbers[i:i+2] for i in range(0, len(numbers), 2)]
                    labels[i][0] = sub_arrays
                    # print(labels[i][0])
                line_image = numpy.zeros((height_standard, width_standard))
                for i in labels:
                    for j in i:
                        j = [[j[t], j[t+1]] for t in range(len(j)-1)]
                        for z in j:
                            cv2.line(line_image, tuple(map(int, z[0])), tuple(map(int, z[1])), color = (255, 255, 255), thickness=5)
                # matplotlib.pyplot.imshow(line_image)
                # matplotlib.pyplot.show()
                line_image = crop_image(line_image)
                line_image = cv2.resize(line_image, (width, height))
                line_image = line_image / 255.0
                cv2.imwrite(file_save_name, line_image)
                y = y + 1
                # print(line_image)

quantityTrain = 20000
quantityVal = 2000

x = 0
y = 0
for file in os.listdir(fileTrain):
    if (bool(quantityTrain) and x == quantityTrain and y == quantityTrain): 
            break
    file_name = os.path.join(fileTrain, file)
    read_data_file(file_name, fileSaveTrain, quantityTrain)

x = 0
y = 0

for file in os.listdir(fileVal):
    if (bool(quantityVal) and x == quantityVal and y == quantityVal): 
            break
    file_name = os.path.join(fileVal, file)
    read_data_file(file_name, fileSaveVal, quantityVal)