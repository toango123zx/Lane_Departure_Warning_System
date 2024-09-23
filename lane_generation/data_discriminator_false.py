import os
import numpy
import cv2
import matplotlib
from PIL import Image
import tensorflow

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

x = 0
y = 0

def read_data_file(path):
    images = []
    for file in os.listdir(path):
        file_name = os.path.join(path, file)
        # print(file_name)
        if (".jpg" in file):
            img = numpy.array(Image.open(file_name))
            images.append(img)
            # matplotlib.pyplot.imshow(img)
            # matplotlib.pyplot.show()
    return images

xImage = read_data_file(fileDataDiscriminatorXTrue)

yImagePredict = []

print(len(xImage))

gen_pix = tensorflow.keras.models.load_model('model_gen_17k_04-22_20-35_0.975.h5')
gen_pix.summary()

length = len(xImage)
xImage1 = xImage[:int(length /2)]
xImage2 = xImage[int(length /2):]

print(len(xImage1) + len(xImage2))

xImage1 = numpy.array(xImage1)
print(xImage1.shape)
xImage2 = numpy.array(xImage2)
print(xImage2.shape)

yImagePredict1 = gen_pix.predict(xImage1)

length = len(yImagePredict1)
length

for i in range(length):
    file_name = os.path.join(fileDataDiscriminatorPredict, f'{i}.jpg')
    cv2.imwrite(file_name, yImagePredict1[i] * 255)

del yImagePredict1
del xImage1

yImagePredict2 = gen_pix.predict(xImage2)
for i in range(len(yImagePredict2)):
    file_name = os.path.join(fileDataDiscriminatorPredict, f'{i + length}.jpg')
    cv2.imwrite(file_name, yImagePredict2[i] * 255)