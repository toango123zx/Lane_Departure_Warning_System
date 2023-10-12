# import numpy
# import os
# import cv2
# import time
# import matplotlib.pyplot
# from PIL import Image
# from tensorflow import keras
# import tensorflow
# from tensorflow.keras import layers
# # from tensorflow.keras import layers
# from tensorflow.keras import models

# width = 160
# height = 60

# # width = 800
# # height = 300

# dict = {'true': [1, 0], 'false': [0, 1]}
# name_result = ['true', 'false']
# # Xtrain = Xtrain/255

# new_model = tensorflow.keras.models.load_model('model_demo_1_10epochs.h5')
# # Check its architecture
# new_model.summary()

# TRAIN_DATA = 'data/lane'

# def DocDuLieu(file):
#     DuLieu = []
#     Label = []
#     label = ''
#     for file in os.listdir(TRAIN_DATA):
#         if (file == 'test'):
#             file_path = os.path.join(TRAIN_DATA, file)
#             list_filename_path = []
#             label = file
#             for filename in os.listdir(file_path):
#                 if (".jpg" in filename or ".png" in filename):
#                     filename_path = os.path.join(file_path, filename)
#                     img = numpy.array(Image.open(filename_path))
#                     img = cv2.resize(img, (width, height))
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                     list_filename_path.append(img)
#                     # Label.append(dict[(label)])
#             DuLieu.extend(list_filename_path)
#     return DuLieu, Label


# Xtrain, Ytrain = DocDuLieu(TRAIN_DATA)
# # for i in range(0, len(Xtrain)):
# #     result = new_model.predict(Xtrain[i].reshape(-1 ,height, width, 1))
# #     # print(reverse_dict[tuple(numpy.array(result[0]))])
# #     print(result)
# # result = new_model.predict(Xtrain[0].reshape(-1 ,height, width, 1))
# # print(result)
# for i in Xtrain:
#     print(name_result[numpy.argmax(new_model.predict(i.reshape(-1, height, width, 1)))])

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

width = 160
height = 60

# width = 800
# height = 300

dict = {'true': [1, 0], 'false': [0, 1]}
name_result = ['right', 'wrong']
# Xtrain = Xtrain/255

new_model = tensorflow.keras.models.load_model('model_demo_1_10epochs.h5')
# Check its architecture
new_model.summary()
    
TRAIN_DATA = 'data/lane'

def DocDuLieu(file):
    DuLieu = []
    Label = []
    label = ''
    for file in os.listdir(TRAIN_DATA):
        if (file == 'test'):
            file_path = os.path.join(TRAIN_DATA, file)
            list_filename_path = []
            label = file
            for filename in os.listdir(file_path):
                if (".jpg" in filename or ".png" in filename):
                    filename_path = os.path.join(file_path, filename)
                    img = numpy.array(Image.open(filename_path))
                    # img = cv2.resize(img, (width, height))
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    list_filename_path.append(img)
                    # Label.append(dict[(label)])
            DuLieu.extend(list_filename_path)
    return DuLieu, Label

Xtrain, Ytrain = DocDuLieu(TRAIN_DATA)
def result(iamge):
    iamge = cv2.cvtColor(iamge, cv2.COLOR_BGR2GRAY)
    result = name_result[numpy.argmax(new_model.predict(iamge.reshape(-1, height, width, 1)))]
    return result

for i in Xtrain:
    print(result(i))
