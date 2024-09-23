import numpy
import os
import cv2
import time
import matplotlib.pyplot
from PIL import Image
from tensorflow import keras
import tensorflow
from tensorflow.keras import layers
# from tensorflow.keras import layers
from tensorflow.keras import models

width_standard= 1640
height_standard = 590
width = 512
height = 128

fileTrain = "./DataTrain"
fileVal = "./DataVal"
fileTest = "./DataTest"
filePredict = "./DataPredict"

def regon_of_interest(image):
    polygons = numpy.array([[(0, height_standard), (0, 300), (700, 240), (850, 240), (width_standard, 300), (width_standard, height_standard)]])
    mask = numpy.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


def crop_image(image):
    return numpy.array(Image.fromarray(image).crop((0, 240, width_standard, height_standard)))

def read_data_file(path):
    images = []
    images_train =[]
    images_label_train = []
    lables_train = []
    print(path)
    for file in os.listdir(path):
        file_name = os.path.join(path, file)
        if (".jpg" in file):
            img = numpy.array(Image.open(file_name))
            images.append(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = regon_of_interest(img)
            img = crop_image(img)
            img = cv2.GaussianBlur(src=img, ksize=(5, 5), sigmaX=0, sigmaY=0)
            img = cv2.resize(img, (width, height))
            img = cv2.Canny(img, 50, 150)
            images_train.append(img)
            # matplotlib.pyplot.imshow(img)
            # matplotlib.pyplot.show()
        if (".txt" in file):
            with open(file_name, 'r') as f:
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
                # images_label_train.append(labels)
                line_image = numpy.zeros((height_standard, width_standard))
                # for i in labels:
                #     for j in i:
                #         for z in j:
                #             cv2.circle(line_image,  tuple(map(int, z)), radius = 5, color = (255, 255, 255), thickness=-1)
                for i in labels:
                    for j in i:
                        # print(j)
                        j = [[j[t], j[t+1]] for t in range(len(j)-1)]
                        # print(j)
                        for z in j:
                            cv2.line(line_image, tuple(map(int, z[0])), tuple(map(int, z[1])), color = (255, 255, 255), thickness=5)
                # matplotlib.pyplot.imshow(line_image)
                # matplotlib.pyplot.show()
                line_image = crop_image(line_image)
                line_image = cv2.resize(line_image, (width, height))
                images_label_train.append(line_image)
                line_image = line_image / 255.0
                # print(line_image)
    return [images, images_train, images_label_train, lables_train]
    
xImage = []
xImgTrain = []
yImgTrain = []
yLabelTrain = []

xImgVal = []
yImgVal = []
yLabelVal = []

for file in os.listdir(fileTest):
    file_name = os.path.join(fileTest, file)
    [xI, xIT, yIT, yLT] = read_data_file(file_name)
    xImage.extend(xI)
    xImgTrain.extend(xIT)
    yImgTrain.extend(yIT)
    yLabelTrain.extend(yLT)

print(len(xImage))
print(len(xImgTrain))
print(len(yImgTrain))
print(len(yLabelTrain))

# new_model = tensorflow.keras.models.load_model('model_demo_1_10epochs.h5')
# # Check its architecture
# new_model.summary()

# result = new_model.predict(xImgTrain[0].reshape(-1, height, width, 1))
# result[0] = result[0] * 255

# matplotlib.pyplot.imshow(xImage[0])
# matplotlib.pyplot.show()
# matplotlib.pyplot.imshow(xImgTrain[0])
# matplotlib.pyplot.show()
# matplotlib.pyplot.imshow(yImgTrain[0])
# matplotlib.pyplot.show()
# matplotlib.pyplot.imshow(result[0])
# matplotlib.pyplot.show()

# cv2.imwrite('image.png', img) 
# Check its architecture
# new_model.summary()