import numpy
import os
import cv2
import time
import pandas
import matplotlib.pyplot
from PIL import Image
from tensorflow import keras
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from scipy.stats import linregress

#non_noise
from sklearn.cluster import DBSCAN

width = 1640
height = 590

Xtrain =[]
Ytrain = []
TRAIN_DATA = 'data/lane'

def DocDuLieu(file):
    DuLieu = []
    Label = []
    label = ''
    for file_path in os.listdir(file):
        if (file_path == 'input'):
            file_path = os.path.join(file, file_path)
            list_filename_path = []
            label = file_path
            for filename in os.listdir(file_path):
                if (".jpg" in filename or ".png" in filename):
                    filename_path = os.path.join(file_path, filename)
                    img = numpy.array(Image.open(filename_path))
                    img = cv2.resize(img, (width, height))
                    list_filename_path.append(img)
                    # Label.append(dict[(label)])
            DuLieu.extend(list_filename_path)
    return DuLieu, Label

def image_canny(lane_image):
    image_gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    # image_gray_blur = cv2.GaussianBlur(image_gray, (7, 7), 2)
    image_gray_blur = cv2.GaussianBlur(src=image_gray, ksize=(5, 5), sigmaX=0, sigmaY=0)
    # image_gray_blur = cv2.medianBlur(src=image_gray, ksize=3)
    image_gray_blur_canny = cv2.Canny(image_gray_blur, 50, 150)
    return image_gray_blur_canny

def regon_of_interest(image):
    polygons = numpy.array([[(535, 415), (1185, 415), (822, 280), (788, 280)]])
    mask = numpy.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    mask_canny = cv2.bitwise_and(image, mask)
    return mask_canny

def display_lines(image, lines):
    line_image = numpy.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return line_image

def make_coordinates(image, line_parameters, location):
    slope, intercept = line_parameters
    if location == 'right':
        y1 = 415
    else:
        y1 = 465
    y2 = 300
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return numpy.array([x1, y1, x2, y2])

#Hàm loại bỏ những điểm nhiễu
def non_noise(lines):
    data = []
    for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            data.append((x1, y1))
            data.append((x2, y2))
    data = numpy.array(data)
    eps = 60.0  # Độ lớn của cửa sổ
    min_samples = 2  # Số điểm tối thiểu trong mỗi cụm
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)

# Lấy các điểm không phải điểm nhiễu
    core_samples_mask = numpy.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    non_noise_points = data[core_samples_mask]
    if (len(non_noise_points) % 2 != 0):
        non_noise_points = non_noise_points[:-1]
        non_noise_points = non_noise_points.reshape((-1, 4))  
    non_noise_points = non_noise_points.reshape((-1, 4))
    return non_noise_points

def average_slope_intercept(image, lines):
    lines = non_noise(lines)
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = numpy.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = numpy.average(left_fit, axis=0)
    right_fit_average = numpy.average(right_fit, axis=0)
    try:    
        left_line = make_coordinates(image, left_fit_average, 'left')
        try:
            right_line = make_coordinates(image, right_fit_average, 'right')
        except Exception:
            return numpy.array([left_line, right_line])
    except Exception as e:
        try:
            right_line = make_coordinates(image, right_fit_average, 'right')
        except Exception as e:
            raise e
    return numpy.array([left_line, right_line])

def crop_output(image):
    image = image[240:400, 450:1250]
    image = cv2.resize(image, (160, 60))
    return image

Xtrain, Yrain = DocDuLieu(TRAIN_DATA)
# print(type(Xtrain[0]))

def lane_detection_image(image):
    lane_image = numpy.copy(image)
    canny = image_canny(lane_image) 
    regon = regon_of_interest(canny)
    lines = cv2.HoughLinesP(image=regon, rho=1, theta=numpy.pi/180, threshold=40, lines=numpy.array([]), minLineLength=5, maxLineGap=5)

    try:
        averaged_lines = average_slope_intercept(image, lines)
        lines = averaged_lines
    except Exception as e:
        # print('loi duong trung binh')
        raise e

    line_image = display_lines(lane_image, lines)
    combo_image = cv2.bitwise_and(image, line_image)
    combo_image =cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    image_output = crop_output(line_image)
    cv2.imwrite(filename='data/lane/acc/right/ringhtline{}.jpg'.format(count), img=image_output)
    # matplotlib.pyplot.imshow(combo_image)
    # matplotlib.pyplot.show()
    
    # matplotlib.pyplot.imshow(combo_image)
    # matplotlib.pyplot.show()
    # images = [image, canny, regon, line_image, combo_image, image_output]
    # captions = ["Ảnh từ camera", "Trích xuất cạnh", "Lấy vùng cần thiết", "Vẽ làn đường", "Ảnh hoàn chỉnh", "Ảnh đầu vào huấn luyến"]
    # fig, axes = matplotlib.pyplot.subplots(2, 3)
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(images[i])
    #     ax.set_title(captions[i], fontsize=16) 
    #     ax.axis('off')
    # matplotlib.pyplot.tight_layout()
    # matplotlib.pyplot.show()

    return image_output

count = 0
for i in Xtrain:
    try:
        file = lane_detection_image(i)
    except Exception as e:
        print(count)
        continue
    count = count + 1
print(len(Xtrain))

def lane_input(image):
    # count = 0
    image = cv2.resize(image, (width, height))
    try:
        image = lane_detection_image(image)
    except Exception as e:
        raise e
    return image